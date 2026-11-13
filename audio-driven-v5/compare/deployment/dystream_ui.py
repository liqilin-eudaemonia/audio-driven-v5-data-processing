from pathlib import Path
import sys,os,argparse
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=20491);args=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'DyStream'))
for name in ['no_proxy','NO_PROXY']:os.environ[name]='localhost,127.0.0.1,'+os.environ.get(name,'')
import app
app.build_ui().queue().launch(server_name='0.0.0.0',server_port=args.port,share=False,show_error=True,css=app.CUSTOM_CSS,theme=app.gr.themes.Soft())
