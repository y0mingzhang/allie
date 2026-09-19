"""Authenticated cluster-local HTTP transport for a single resident GPU oracle."""
import io,json,secrets,os
from http.server import BaseHTTPRequestHandler,HTTPServer
import numpy as np

def make_server(forward,out):
    token_path=out/'rpc-token'
    if not token_path.exists():
        fd=os.open(token_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(32))
    token=token_path.read_text().strip()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if self.headers.get('Authorization')!='Bearer '+token:
                self.send_error(403);return
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=32*1024*1024:self.send_error(413);return
            try:
                r=json.loads(self.rfile.read(size));pred=forward(r['prefixes'])
                if 'columns' in r:pred=pred[:,r['columns']]
                buf=io.BytesIO();np.save(buf,pred,allow_pickle=False);body=buf.getvalue()
                self.send_response(200);self.send_header('Content-Type','application/octet-stream')
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
            except Exception as e:
                self.send_error(500,str(e))
    server=HTTPServer(('0.0.0.0',0),Handler);server.timeout=.05
    return server
