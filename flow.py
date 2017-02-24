import numpy as np 
from prac import *
import mnist as mn 


dataset=mn.read("training")

labels=np.array([[1,0,0,0,0,0,0,0,0,0],
				 [0,1,0,0,0,0,0,0,0,0],
				 [0,0,1,0,0,0,0,0,0,0],
				 [0,0,0,1,0,0,0,0,0,0],
				 [0,0,0,0,1,0,0,0,0,0],
				 [0,0,0,0,0,1,0,0,0,0],
				 [0,0,0,0,0,0,1,0,0,0],
				 [0,0,0,0,0,0,0,1,0,0],
				 [0,0,0,0,0,0,0,0,1,0],
				 [0,0,0,0,0,0,0,0,0,1]])

fils1no=3  
fils2no=2 
hiddenno=15     
batch=30
lr=0.1

np.random.seed(1)
#initialize weights with mean of zero
fils1=2*np.random.random((fils1no,1,3,3))-1
w0=2*np.random.random((13*13*fils1no,hiddenno))-1
w1=2*np.random.random((hiddenno,10))-1
bias1=2*np.random.random((fils1no,1))-1
b0=2*np.random.random((1,hiddenno))-1
b1=2*np.random.random((1,10))-1

gfils1=np.zeros(fils1.shape)
gw0=np.zeros(w0.shape)
gw1=np.zeros(w1.shape)
gbias1=np.zeros(bias1.shape)
gb0=np.zeros(b0.shape)
gb1=np.zeros(b1.shape)

batchno=0

step=0
#forward and backward pass
for epoch in range(300):
	count=0
	print("epoch"+str(epoch)+"#############################################")
	dataset=mn.read("training",".")
	c=0
	if epoch%10==0:
		np.save("./weights/fils1",fils1)
		np.save("./weights/bias1",bias1)
		np.save("./weights/w0",w0)
		np.save("./weights/w1",w1)
		np.save("./weights/b0",b0)
		np.save("./weights/b1",b1)

	for data in dataset:
		if True:
			c+=1
			image=shapeprop(data[1],"i")/255
			

			label=labels[data[0]]


			act1=convolve(image,fils1,bias1)
			act2,ind2=maxpool(act1)
			h,d,w=act2.shape
			act5=act2.reshape((1,h*d*w))
			
			#act5=image.reshape((1,784))
			act6=sigmoid(np.dot(act5,w0)+b0)
			act7=sigmoid(np.dot(act6,w1)+b1)
			
			error=act7-label

			del7=error*sigmoidprime(act7)
			del6=del7.dot(w1.T)*sigmoidprime(act6)
			del5=del6.dot(w0.T)*sigmoidprime(act5)
			del2=del5.reshape(act2.shape)
			del1=delpooltoconv(del2,ind2)
			
			bias1grad,fil1grad=gradoffilter2(del1,image)
			w0grad=act5.T.dot(del6)
			w1grad=act6.T.dot(del7)
			b0grad=del6
			b1grad=del7
			
			gfils1-=fil1grad
			gbias1-=bias1grad
			gw0-=w0grad
			gw1-=w1grad
			gb0-=b0grad
			gb1-=b1grad


			batchno+=1
			if batchno%1==0:
				fils1+=lr*gfils1
				w0+=lr*gw0
				w1+=lr*gw1
				bias1+=lr*gbias1
				b0+=lr*gb0
				b1+=lr*gb1

				gfils1=np.zeros(fils1.shape)
				gw0=np.zeros(w0.shape)
				gw1=np.zeros(w1.shape)
				gbias1=np.zeros(bias1.shape)
				gb0=np.zeros(b0.shape)
				gb1=np.zeros(b1.shape)


			if data[0]==np.argmax(act7):
				count+=1
			step+=1
			if step%100==0:
				print("step :"+str(step)+"   LOSS:  "+str(np.mean(np.abs(error)))+" label"+str(data[0]))
			if c==1000:
				step=0
				st=0
				print("accuracy:"+str(count))
				correct=0	
				testing=mn.read("testing",".")
				for test in testing:
					image=shapeprop(test[1],"i")
					image=image/255
					label=labels[test[0]]
					st+=1
					act1=convolve(image,fils1,bias1)
					act2,ind2=maxpool(act1)
					h,w,d=act2.shape
					act5=act2.reshape((1,h*d*w))
					act6=sigmoid(np.dot(act5,w0)+b0)
					act7=sigmoid(np.dot(act6,w1)+b1)
					if test[0]==np.argmax(act7):
						correct+=1
					if st==100:
						print("test accuracy->"+str(correct)) 
						break				
				break
					



						











