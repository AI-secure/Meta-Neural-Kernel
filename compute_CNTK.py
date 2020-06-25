from support.ConvNTK import CNTK_value
import numpy as np
import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=str, help='id', default='0')
argparser.add_argument('--n-thread', type=int, default=6)
argparser.add_argument('--id', type=int, default=0)
argparser.add_argument('--save-freq',type=int,default=300000)
argparser.add_argument("--load-kernel", default=False, action="store_true" , help="Flag to do something")
argparser.add_argument("--load", default=False, action="store_true" , help="Flag to do something")
argparser.add_argument("--reverse", default=False, action="store_true" , help="Reverse order")
argparser.add_argument('--path',default='saved_models/CNTK_elements-200/',type=str)
argparser.add_argument('--dim',default=-1,type=int,help='dimension of the CNTK. i.e. the N of the shape (N,N). -1 means no limit')
args = argparser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
print("Set GPU = %s"%args.gpu)
d = 20
fix = False
gap = True
X = np.load('saved_models/X-200.npy')
print("X.shape",X.shape)

if args.load:
    print("Load from %s"%args.path)
CNTK_value(X=X,d=d,fix=fix,gap=gap,multi_thread=True,n_thread=args.n_thread,thread_id=args.id,
           save_path=args.path,save_freq=args.save_freq,load=args.load,load_kernel=args.load_kernel,reverse_order=args.reverse,dim=args.dim)
