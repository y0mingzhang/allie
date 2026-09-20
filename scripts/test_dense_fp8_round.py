"""Eager/compiled quantizer bytes against direct CUDA RNE, including all finite midpoints."""
import argparse,json
import torch
from modded_fp8 import round_e4m3,quantize

def main():
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');a=p.parse_args()
 torch.manual_seed(71)
 grid=torch.arange(256,dtype=torch.uint8).view(torch.float8_e4m3fn).float()
 grid=grid[torch.isfinite(grid)].sort().values.unique()
 mid=(grid[:-1]+grid[1:])/2
 v=torch.cat((torch.rand(131072)*896-448,grid,mid,
    torch.nextafter(mid,torch.full_like(mid,float('inf'))),
    torch.nextafter(mid,torch.full_like(mid,-float('inf'))),
    torch.tensor([-0.,0.,12.5065,400.13,-116.0013])))
 # torch eager CUDA itself is the correctly-rounded conversion oracle.
 v=v.to(a.device);want=v.to(torch.float8_e4m3fn).view(torch.uint8)
 reports=[]
 for compile in (False,True):
  fn=lambda x:round_e4m3(x).to(torch.float8_e4m3fn)
  if compile:fn=torch.compile(fn,fullgraph=True)
  got=fn(v).view(torch.uint8)
  n=int((got!=want).sum());assert n==0,(compile,n)
  q=torch.compile(quantize,fullgraph=True) if compile else quantize
  x=torch.cat((v,torch.tensor([-448.,448.],device=a.device)))
  out,scale=q(x)
  assert float(scale)==1.
  assert torch.equal(out.view(torch.uint8),x.to(torch.float8_e4m3fn).view(torch.uint8))
  reports.append(dict(compiled=compile,values=v.numel(),byte_differences=n))
 print(json.dumps(reports),flush=True)
if __name__=='__main__':main()
