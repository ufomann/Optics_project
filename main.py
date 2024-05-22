import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
import os

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
 
w = 1500
h = 1000
 
names = [8859, 8863, 8874, 8945, 8979, 9048, 9150, 9265, 9289]

for name in names:
    os.system(f'mkdir ./data/IMG_{name}')

 
for name in names:
    name = str(name)
    img = np.array(Image.open('./data/IMG_'+name+'.JPG').convert('L').resize((w,h)))

    l = 0.000633 #wavelength, mm

    for z in range(0,5000,100):
        x = np.linspace(-22.3/2,22.3/2,w) #photo sensor width, mm
        y = np.linspace(-14.9/2,14.9/2,h) #photo sensor height, mm
 
        xx, yy = np.meshgrid(x, y)

        fx = np.fft.fftfreq(x.shape[-1],d=22.3/w)
        fy = np.fft.fftfreq(y.shape[-1],d=14.9/h)
 
        fxx, fyy = np.meshgrid(fx, fy)
 
        def H(z):
            return np.exp((2*np.pi/l)*z*np.sqrt(1-(l**2)*(fxx**2+fyy**2))*1j)

        G0 = abs(img)*np.exp(np.random.uniform(0, 2*np.pi, (h,w))*1j)
 
        for i in range(5):
            print(i)
            gn = np.fft.ifft2(np.fft.fft2(G0)*H(-z))
 
            grad_gn = np.gradient(gn)
            grad_gn_abs = abs(grad_gn[0])**2+abs(grad_gn[1])**2
            delta = (np.sqrt(np.sum((grad_gn_abs)**2)))*100
            grad_gn = grad_gn/np.sqrt(1+grad_gn_abs/(delta**2))
            grad_gn_xx = np.gradient(grad_gn[0])
            grad_gn_yy = np.gradient(grad_gn[1])
            gradH = grad_gn_xx[0]+grad_gn_yy[1]
 
            u = gradH/np.sqrt(np.sum(abs(gradH)))
 
            for j in range(1):
                gn = gn + 0.001*u*np.sqrt(np.sum(abs(gn)))
 
            Uz = np.fft.ifft2(np.fft.fft2(abs(gn))*H(z))
            G0 = abs(img)*np.exp(np.angle(Uz)*1j)
 
        gn = np.fft.ifft2(np.fft.fft2(G0)*H(-z))
        res = Image.fromarray(abs(gn)).convert("L")
        res.save('./data/IMG_'+name+'/'+str(z)+'.bmp')
