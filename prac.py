import numpy as np 
from PIL import Image

def rot180(mat):
	"""returns mat with each filter rotated by 180 degrees spacially"""
	mat=np.array([np.rot90(m,2) for m in mat])
	return mat


def shapeprop(filter,ftype):
	"""converts 2d to 3d or 3d to 4d"""
	if ftype=="f":
		depth,height,width=filter.shape
		return filter.reshape((depth,1,height,width))
	else:
		height,width=filter.shape
		return filter.reshape((1,height,width))

def zeropad(mat):
	"""zeropads the matrix mat"""
	depth,height,width=mat.shape
	zer=np.zeros((depth,height+2,width+2))
	zer[:,1:height+1,1:width+1]=mat
	return zer

def convolve(inp,filter,bias,stride=1):
	"""inp is 3d array
	filter is a 4d array constisting of many 3d filters"""
	#print("filter in convolve")
	#print(filter)
	dep,h,w=inp.shape
	filnum,depth,fh,fw=filter.shape
	od=(h-fh)/stride+1
	tot=int(od*od)
	output=np.zeros((filnum,tot))
	#print("bais shape:"+str(bias.shape))
	
	for d in range(filnum): 
		row=0
		slide=0
		for i in range(tot):
			inpart=inp[:,row:fw+row,slide:fw+slide]
			
			output[d][i]=np.sum(inpart*filter[d])+bias[d]
			
			slide+=stride
			if slide==od:
				slide = 0
				row+=stride
				if row==od:
					break

	output=output.reshape((filnum,int(od),int(od)))
	return sigmoid(output)

def deltoconv(deltasmall,filters):
	""" """
	"""deltasmall is 3d array
	filter is 4d array"""
	nf,d,h,w=filters.shape
	for i in range(w-1):
		deltasmall=zeropad(deltasmall) 
	depth,height,width=deltasmall.shape
	od=int((height-h)/1+1)
	delbig=np.zeros((d,od,od))
	for dlayer,flayer in zip(deltasmall,filters):
		#flayer=rot180(flayer)
		dlayer=shapeprop(dlayer,"d")
		flayer=shapeprop(flayer,"f")
		no,d,h,w=flayer.shape
		delbig+=convolve(dlayer,flayer,np.zeros((flayer.shape[0],1)))
	return delbig


def deltoconv2(deltasmall,filters):
	nf,df,hf,wf=filters.shape
	depth,height,width=deltasmall.shape
	deltasmall=deltasmall.reshape((depth,height*width))
	od=hf+height-1
	delbig=np.zeros((df,od,od))
	# print(delbig.shape)
	for d in range(nf):
		row=0
		slide=0
		for i in range(height*width):
			#print(i)
			#print(deltasmall[d][i])
			#print(delbig[:,row:row+hf,slide:slide+hf].shape)
			delbig[:,row:row+hf,slide:slide+hf]+=filters[d]*deltasmall[d][i]

			slide+=1
			if slide==od-2:
				slide=0
				row+=1
				if row==od-2:
					break

	return delbig




def maxpool(inp):
	"""performs pooling operation
	inp is 3d array"""
	
	depth,height,width=inp.shape
	"""print("inp.shape")
	print(inp.shape)"""
	poolsize=int(height/2)*int(width/2)
	

	output=np.zeros((depth,poolsize))
	maxindex=np.zeros((depth,poolsize,2))
	
	
	for d in range(depth):
		row=0
		slide=0
		for i in range(poolsize):
			sam=inp[d][row:row+2,slide:slide+2]
			"""print("sam")
			print(sam)"""
			output[d][i]=np.amax(sam)
			index=[ind for ind in zip(*np.where(sam==np.max(sam)))]
			if len(index)>1:
				index=[index[0]]
			maxindex[d][i]=index[0][0]+row,index[0][1]+slide
			
			slide+=2
			if slide>=(int(height)):
				slide = 0
				row+=2
				
	output=output.reshape((depth,int(height/2),int(width/2)))
	maxindex=maxindex.reshape((depth,int(height/2),int(width/2),2))
	return [output,maxindex]

def delpooltoconv(pooldelta,maxindex):
	""""""
	depth,height,width=pooldelta.shape
	pooldelta=pooldelta.reshape((depth,height*width))
	convdeltas=np.zeros((depth,height*2,width*2))
	maxindex=maxindex.reshape((depth,height*width,2))
	for d in range(depth):
		for i in range(height*width):
			x,y=map(int,maxindex[d][i])
			convdeltas[d][x,y]=pooldelta[d][i]
	return convdeltas


def makeeven(mat):
	depth,height,width=mat.shape
	output=np.zeros((depth,height+1,width+1))
	output[:,:height,:width]=mat
	return output

def gradoffilter2(delta,prevact):
	depth,height,width=prevact.shape
	noofdel,dh,dw=delta.shape
	od=(height-dh) +1
	gradient=np.zeros((noofdel,depth,od*od))
	gradbias=np.zeros((noofdel,1))
	for d in range(noofdel):
		row=0
		slide=0
		for i in range(od*od):
			inp=(prevact[:,row:row+dh,slide:slide+dw])
			xx=inp*delta[d]
			#print("xx shape:"+str(xx.shape))
			gradient[d][:,i]=np.array([np.sum(x) for x in xx])
			slide+=1
			if slide==od:
				slide=0
				row+=1
		gradbias[d]=np.sum(delta[d])   

	return gradbias,gradient.reshape((noofdel,depth,od,od))




def sigmoid(inp):
	return 1.0/(1.0+np.exp(-inp))


def sigmoidprime(x):
	return x*(1-x)

def relu(x):
	return np.maximum(0,x)

def reluprime(x):
	return np.ones(x.shape)*(x>0)

