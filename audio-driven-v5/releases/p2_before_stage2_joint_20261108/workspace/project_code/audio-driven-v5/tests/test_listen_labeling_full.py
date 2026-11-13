import base64,copy,io,json,sys,tempfile,unittest
from pathlib import Path
import av,numpy as np,soundfile as sf
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from listen_labeling.full_source import compact_source,prepare_source
from p2_listen.contracts import sha

class FullSourceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);video=self.root/'video.mp4'
  with av.open(str(video),'w') as c:
   s=c.add_stream('mpeg4',rate=20);s.width=s.height=64;s.pix_fmt='yuv420p'
   for i in range(40):
    pixels=np.zeros((64,64,3),dtype=np.uint8);pixels[:,:32,0]=240;pixels[:,32:,2]=220
    for p in s.encode(av.VideoFrame.from_ndarray(pixels,format='rgb24')):c.mux(p)
   for p in s.encode():c.mux(p)
  source=self.root/'source.bin';source.write_bytes(b'PREFIX123456'+video.read_bytes()+b'END')
  ref=dict(archive=str(source),member='listener.mp4',offset=12,byte_size=video.stat().st_size)
  wav=self.root/'listener.wav';wave=.2*np.sin(np.arange(32000)*.07);sf.write(wav,wave,16000,subtype='PCM_24')
  self.row=dict(sample_id='source_test',pair='pair',split='train',listener_id='listener',speaker_id='speaker',listener_index=1,speaker_index=0,
   admission_pool='primary',audio_final=dict(pool='keep_candidate'),sources=dict(listener=ref,speaker=dict(member='DO_NOT_READ')),
   normalized_audio=dict(listener=dict(path=str(wav),sha256=sha(wav),sample_rate=16000,contract=dict(source=ref))),
   face_geometry=dict(listener=dict(width=64,height=64)),crop_proposals=dict(listener=dict(status='proposal_only',x=32,y=0,size=32)),start=.4,end=1.2,duration_s=.8)
  self.cfg=dict(fps=8,max_frames=128,max_audio_seconds=16)
 def tearDown(self):self.tmp.cleanup()
 def test_nonzero_source_cut_uses_listener_pixels_audio_and_local_pts(self):
  row=compact_source(self.row,'manifest.jsonl',1);value=prepare_source(row,self.cfg)
  self.assertAlmostEqual(value['duration_s'],.8)
  content,meta=value['visual'];self.assertTrue(all(0<=f['time_s']<.8 for f in meta['video']['frames']))
  im=Image.open(io.BytesIO(base64.b64decode(content[1]['image_url']['url'].split(',')[1])))
  self.assertEqual(im.size,(512,512));pixel=np.asarray(im).mean(axis=(0,1));self.assertGreater(pixel[2],180);self.assertLess(pixel[0],30)
  payload=value['audio'][0][1]['input_audio']['data'];actual,rate=sf.read(io.BytesIO(base64.b64decode(payload)))
  expected,_=sf.read(self.row['normalized_audio']['listener']['path'],start=6400,stop=19200)
  self.assertEqual(rate,16000);np.testing.assert_allclose(actual,expected,atol=1/32768)
  self.assertEqual(value['audio'][1]['audio']['source_start_sample'],6400)
 def test_mixed_role_source_rejected_before_any_decoding(self):
  r=copy.deepcopy(self.row);r['normalized_audio']['listener']['contract']['source']['member']='speaker.mp4'
  with self.assertRaisesRegex(ValueError,'provenance'):compact_source(r,'manifest',1)
 def test_unavailable_source_audio_interval_not_padded(self):
  r=copy.deepcopy(self.row);r.update(start=1.8,end=2.3,duration_s=.5)
  with self.assertRaisesRegex(ValueError,'outside source'):prepare_source(compact_source(r,'manifest',1),self.cfg)
 def test_rounded_metadata_duration_is_tolerated_but_material_change_rejected(self):
  r=copy.deepcopy(self.row);r['duration_s']+=.000001;compact_source(r,'manifest',1)
  r['duration_s']+=.01
  with self.assertRaisesRegex(ValueError,'duration'):compact_source(r,'manifest',1)
 def test_shard_resume_does_not_duplicate_success_or_call_models_again(self):
  import importlib.util,time,threading
  from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
  spec=importlib.util.spec_from_file_location('batch_runner',Path(__file__).resolve().parents[1]/'scripts/run_listen_labeling_full.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  calls=[]
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_POST(self):
    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));calls.append(body)
    text=body['messages'][0]['content'][0]['text'];visual='silent timestamped' in text;discovery='Task: open_discovery' in text
    key=('visual' if visual else 'audio') if discovery else ('head_nod' if visual else 'listener_vocalization')
    value=dict(reviewed={key:[[0,.8]]},unknown={key:[]},events=[],notes='MOCK_TEST_ONLY')
    self.send_response(200);self.end_headers();self.wfile.write(json.dumps(dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(value)))])).encode())
  server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   root=self.root/'batch';out=root/'production'
   for name in ['locks','state','results','inflight']:(out/name).mkdir(parents=True)
   (root/'queue'/'shards').mkdir(parents=True);row=compact_source(self.row,'fixture',1)
   (root/'queue'/'shards'/'000.jsonl').write_text(json.dumps(row)+'\n')
   ep=dict(base_url=f'http://127.0.0.1:{server.server_port}/v1',model='mock',max_tokens=512,timeout_s=10)
   cfg=dict(revision='fixture',media=self.cfg,guided_json=True,model_revisions={'visual':'mock','audio':'mock'},endpoints={'visual':[ep],'audio':[ep]})
   (root/'production_config.json').write_text(json.dumps(cfg));module.worker(str(root),0,1,time.time()+30)
   state=json.loads((out/'state'/'000.json').read_text());self.assertEqual(state['processed'],1);self.assertEqual(state['schema_pass'],1);self.assertEqual(len(calls),4)
   self.assertTrue(all('response_format' in call for call in calls))
   # Simulate a crash after atomic result commit, before progress commit.
   for key in ['offset','processed','schema_pass','failed','supervision_supported']:state[key]=0
   state['done']=False;(out/'state'/'000.json').write_text(json.dumps(state));module.worker(str(root),0,1,time.time()+30)
   self.assertEqual(len(calls),4);self.assertEqual(json.loads((out/'state'/'000.json').read_text())['processed'],1)
  finally:server.shutdown();server.server_close();thread.join()
 def test_invalid_semantics_requires_new_model_response_not_label_repair(self):
  import importlib.util
  spec=importlib.util.spec_from_file_location('batch_retry',Path(__file__).resolve().parents[1]/'scripts/run_listen_labeling_full.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  class Response:
   status_code=200
   def __init__(self,value):self.text=json.dumps(dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(value)))]))
   def json(self):return json.loads(self.text)
   def raise_for_status(self):pass
  class Session:
   def __init__(self):self.calls=[]
   def post(self,url,json,timeout):
    self.calls.append(copy.deepcopy(json));return Response(dict(reviewed={'listener_vocalization':[[.7,.2]] if len(self.calls)==1 else [[0,.8]]},unknown={'listener_vocalization':[]},events=[],notes='MOCK'))
  session=Session();media=[dict(type='text',text='ORIGINAL_LISTENER_MEDIA_MARKER')]
  result=module.invoke(session,dict(model='mock',base_url='http://localhost/v1'),'audio','audit',.8,media,True)
  self.assertEqual(result['status'],'schema_pass');self.assertEqual(len(result['attempts']),2);self.assertIn('validation_error',result['attempts'][0]);self.assertNotIn('error',result)
  self.assertEqual(session.calls[0]['messages'][0]['content'][1:],session.calls[1]['messages'][0]['content'][1:])
  self.assertNotEqual(result['attempts'][0]['request_sha256'],result['attempts'][1]['request_sha256'])
if __name__=='__main__':unittest.main()
