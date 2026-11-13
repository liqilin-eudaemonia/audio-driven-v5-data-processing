import copy,unittest
import run as a

class PreparedTests(unittest.TestCase):
 def test_scope_and_checkpoints(self):
  combos=a.plans();self.assertEqual(len(combos),12);self.assertEqual(len(a.slots()),24)
  for component,last in [('no_event',6573),('no_feature',6643),('no_whisper',6643)]:
   self.assertEqual(sorted(x['ltx'] for x in combos if x['component']==component),[5500,6000,6500,last])
  self.assertTrue(all(x['qwen']==7500 for x in combos))
 def test_identical_source_conditions_and_protocol(self):
  old=a.ROOT/'experiments/p2_joint_qwen_ltx_test64_16gpu_20261113'
  baseline=a.read(old/'combinations/qwen7500_ltx6800/protocol.json');items=a.read(a.WORK/'shared_items.json')
  self.assertEqual(items,a.read(old/'shared_items.json'))
  for x in a.plans():
   w=a.child(x['key']);p=a.read(w/'protocol.json')
   for key in ['generation','seed','datasets','planner_adapter_identity','sampling']:
    self.assertEqual(p[key],baseline[key],(x['key'],key))
   self.assertEqual((w/'conditions').resolve(),(a.cache_dir(7500)/'conditions').resolve())
   shards=[y['key'] for i in range(4) for y in a.read(w/'shards'/f'{i:02d}.json')]
   self.assertEqual(len(shards),64);self.assertEqual(set(shards),{y['key'] for y in items})
 def test_disabled_components_and_no_whisper_cache(self):
  expected={'no_event':dict(event=False,feature=True,whisper=True),'no_feature':dict(event=True,feature=False,whisper=True),'no_whisper':dict(event=True,feature=True,whisper=False)}
  for x in a.plans():
   w=a.child(x['key']);p=a.read(w/'protocol.json');self.assertEqual(p['component_flags'],expected[x['component']])
   self.assertEqual((w/'whisper').exists(),x['component']!='no_whisper')
   identity=p['checkpoint_identity'];self.assertEqual(identity['experiment']['component_arm'],x['component'])
 def test_native_configs_reject_cross_arm(self):
  from p2_whisper_av.bootstrap import activate_ltx
  activate_ltx()
  from p2_ltx_all_lora_components.config import validate
  info=a.read(a.WORK/'PLAN.json')['identities'];c=copy.deepcopy(info['no_whisper']['normalized_config']);validate(c)
  c['component_arm']='no_feature'
  with self.assertRaises(ValueError):validate(c)
 def test_frozen_entry_and_marker_identity(self):
  a.verify()
  for r in a.read(a.WORK/'CHECKPOINTS.json'):
   marker=a.read(a.Path(r['path']).with_suffix('.complete.json'))
   self.assertEqual(marker['step'],r['step']);self.assertEqual(marker['sha256'],r['sha256']);self.assertEqual(a.base.stamp(r['path']),r['stamp'])

if __name__=='__main__':unittest.main()
