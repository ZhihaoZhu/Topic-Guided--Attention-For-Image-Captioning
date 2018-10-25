#import os
#import shutil
#
#oldname="./image/COCO_val2014_000000081264.jpg"
#
#newname= "./COCO_val2014_000000081264.jpg"
#
#shutil.copyfile(oldname,newname)
import shutil
import os

txtName = "./zzzz/caption.txt"
f=file(txtName, "a+")
f.write('1')
f.close()



#import numpy as np
#import tensorflow as tf
#
#xz1=[[0.9],[0.7]]
#
#xz=[0.9,0.7]
#A = [0.8,0.6,0.3]
#B = [0.9,0.5,0.7]
#C = [A,B]
#C = tf.convert_to_tensor(C)
#k=2
#a = {}
#with tf.Session() as sess:
#
#    
#    out,xx= tf.nn.top_k(C[0], 1)
#    print out
#    sess.run(tf.initialize_all_variables())
#    print sess.run(out)
#    print sess.run(xx)
#with tf.Session() as sess:
#    for i in range(k):
#    
#        out,xx= tf.nn.top_k(C[i], k)
#        print out
#        sess.run(tf.initialize_all_variables())
#        print sess.run(out[0])
#
#        for j in range(k):
#            if j==0:
#                a[xz[i]]=[list([out[j],xx[j]])]
#            else:
#                a[xz[i]].append(list([out[j],xx[j]]))
#
#    aaa = sess.run(a)
##    print (aaa)
##    for key,value in aaa.items():
##        print ('key is %s,value is %s'%(key,value))
#
##    print (aaa[xz[0]][0][0])
##    print (aaa[xz[1]])
#
#    sort = []
#    sor = {}
#    for x in range(k):
#        for y in range(k):
#            sort.append(xz[x]*(aaa[xz[x]][y][0]))
#            #sor[aaa[xz[x]][y][1]]=xz[x]*(aaa[xz[x]][y][0])
#            sor[xz[x]*(aaa[xz[x]][y][0])]=[xz[x],aaa[xz[x]][y][1]]
#    for key,value in sor.items():
#        print key
#    xzz = sorted(sor.items(), key=lambda item:item[0],reverse=True)# sorted return a list[]
#    print xzz
#    for qq in range(len(xz1)):
#        for kk in range(k):
#            if xz1[qq][0] == xzz[kk][1][0]:
#                xz1[qq].append(xzz[kk][1][1])
#    print xz1
