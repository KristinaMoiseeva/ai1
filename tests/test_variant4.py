import numpy as np
from scripts.classifiers import KNearestNeighbor, Softmax
from scripts.classifiers.softmax import softmax_loss_naive, softmax_loss_vectorized

def test_knn_distances_match():
    rng=np.random.default_rng(1);X=rng.normal(size=(8,5));Q=rng.normal(size=(3,5));k=KNearestNeighbor();k.train(X,np.arange(8)%2)
    a=k.compute_distances_two_loops(Q);b=k.compute_distances_one_loop(Q);c=k.compute_distances_no_loops(Q)
    np.testing.assert_allclose(a,b);np.testing.assert_allclose(a,c)

def test_knn_predict_shape():
    X=np.array([[0.],[1.],[10.],[11.]]);y=np.array([0,0,1,1]);k=KNearestNeighbor();k.train(X,y)
    assert np.array_equal(k.predict(np.array([[.2],[10.2]]),k=1),np.array([0,1]))

def test_softmax_naive_vectorized_match():
    rng=np.random.default_rng(2);W=rng.normal(scale=.01,size=(6,3));X=rng.normal(size=(7,6));y=rng.integers(0,3,size=7)
    l1,g1=softmax_loss_naive(W,X,y,.1);l2,g2=softmax_loss_vectorized(W,X,y,.1)
    np.testing.assert_allclose(l1,l2,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(g1,g2,rtol=1e-10,atol=1e-12)

def test_softmax_train_predict():
    X=np.array([[2.,0.,1.],[1.,0.,1.],[0.,2.,1.],[0.,1.,1.]])
    y=np.array([0,0,1,1]);m=Softmax(random_state=3);h=m.train(X,y,learning_rate=.1,reg=0,num_iters=200,batch_size=4)
    assert len(h)==200;assert np.mean(m.predict(X)==y)>=.75
