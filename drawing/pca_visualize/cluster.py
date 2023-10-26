import os
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

add='pca'
name='PCA'

YELLOW12= '#DAB47F'
BLUE1= '#A0BBDA'
RED2= '#DAA6C3'
GREEN2= '#609E6F'

DPI=500
SIZE=3
NSTD=2.5
ALPHA= 0.2

feature_bn=np.load('feature_bn_{}.npy'.format(add))
feature_an=np.load('feature_an_{}.npy'.format(add))
label_bn=np.load('4KwPtest_train_85_embedding_label.npy')


def plot_point_cov(points, nstd=3, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)
 
def plot_cov_ellipse(cov, pos, nstd=3, ax=None, **kwargs):
    def eigsorted(cov):
        cov = np.array(cov)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]
 
    vals, vecs = eigsorted(cov)
 
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)




if 1:
    fig = plt.figure(num=2, figsize=(10,4),dpi=DPI)

    mask1=(label_bn>1)
    mask2=(label_bn<=1)

    ax = fig.add_subplot(1,2,1)

    pts=feature_bn[mask2,:]
    newx,newy=pts[:,0],pts[:,1]
    ax.set_title('ACE2 Binding: {} Before Feature Coupling'.format(name),fontdict={'size':12,'family':'Arial','weight':'bold'})
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=BLUE1)
    ax.scatter(newx,newy,s=SIZE,c=BLUE1,label='Harmful mutations',zorder=1)


    pts=feature_bn[mask1,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=YELLOW12)
    ax.scatter(newx,newy,s=SIZE,c=YELLOW12,label='Beneficial mutations',zorder=2)

    plt.xticks((),fontproperties='Arial')
    plt.yticks((),fontproperties='Arial')
    ax.legend(prop={'family':'Arial','size':10,'weight':'bold'},loc=2)


    ax = fig.add_subplot(1,2,2)

    pts=feature_an[mask2,:]
    newx,newy=pts[:,0],pts[:,1]
    ax.set_title('ACE2 Binding: {} After Feature Coupling'.format(name),fontdict={'size':12,'family':'Arial','weight':'bold'})
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=BLUE1)
    ax.scatter(newx,newy,s=SIZE,c=BLUE1,label='Harmful mutations',zorder=1)


    pts=feature_an[mask1,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=YELLOW12)
    ax.scatter(newx,newy,s=SIZE,c=YELLOW12,label='Beneficial mutations',zorder=2)

    plt.xticks((),fontproperties='Arial')
    plt.yticks((),fontproperties='Arial')
    ax.legend(prop={'family':'Arial','size':10,'weight':'bold'},loc=2)

    #plt.show()
    fig.savefig('4k_{}2.png'.format(add))
    fig.clear()


if 1:
    fig = plt.figure(num=2, figsize=(10,4),dpi=DPI)

    mask1=(label_bn>1)
    mask2=(label_bn>0.5)&(label_bn<=1)
    mask3=(label_bn<=0.5)

    ax = fig.add_subplot(1,2,1)

    pts=feature_bn[mask3,:]
    newx,newy=pts[:,0],pts[:,1]
    ax.set_title('ACE2 Binding: {} Before Feature Coupling'.format(name),fontdict={'size':12,'family':'Arial','weight':'bold'})
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=RED2)
    ax.scatter(newx,newy,s=SIZE,c=RED2,label='Harmful mutations (<0.5)',zorder=1)

    pts=feature_bn[mask2,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=GREEN2)
    ax.scatter(newx,newy,s=SIZE,c=GREEN2,label='Harmful mutations (0.5-1)',zorder=2)

    pts=feature_bn[mask1,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=YELLOW12)
    ax.scatter(newx,newy,s=SIZE,c=YELLOW12,label='Beneficial mutations',zorder=3)

    plt.xticks((),fontproperties='Arial')
    plt.yticks((),fontproperties='Arial')
    ax.legend(prop={'family':'Arial','size':10,'weight':'bold'},loc=2)


    ax = fig.add_subplot(1,2,2)

    pts=feature_an[mask3,:]
    newx,newy=pts[:,0],pts[:,1]
    ax.set_title('ACE2 Binding: {} After Feature Coupling'.format(name),fontdict={'size':12,'family':'Arial','weight':'bold'})
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=RED2)
    ax.scatter(newx,newy,s=SIZE,c=RED2,label='Harmful mutations (<0.5)',zorder=1)

    pts=feature_an[mask2,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=GREEN2)
    ax.scatter(newx,newy,s=SIZE,c=GREEN2,label='Harmful mutations (0.5-1)',zorder=2)

    pts=feature_an[mask1,:]
    newx,newy=pts[:,0],pts[:,1]
    plot_point_cov(pts, nstd=NSTD, ax=ax, alpha=ALPHA, color=YELLOW12)
    ax.scatter(newx,newy,s=SIZE,c=YELLOW12,label='Beneficial mutations',zorder=3)

    plt.xticks((),fontproperties='Arial')
    plt.yticks((),fontproperties='Arial')
    ax.legend(prop={'family':'Arial','size':10,'weight':'bold'},loc=2)

    #plt.show()
    fig.savefig('4k_{}3.png'.format(add))