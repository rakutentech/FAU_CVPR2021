import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from keras import backend as K
import numpy as np
from config import AU_count
def get_centers(y_true,y_pred,centers):
        output_list = []

        for i in range(AU_count):
            labels = tf.reshape(y_true[:,i],[-1])
            labels = tf.cast(labels,tf.float32)
            masked_features = centers[i,...]+y_pred[:,i,:]*labels[:,tf.newaxis]
            moving_average = tf.reduce_mean(masked_features,axis=0)
            output_list.append(moving_average)

        centers = tf.stack(output_list) 
        
        return centers

def get_center_loss(alpha,y_true,y_pred,centers):
    loss=0
    for i in range(AU_count):
        labels = tf.reshape(y_true[:,i],[-1])
        labels = tf.cast(labels,tf.float32)
        diff = alpha[i]*(y_pred[:,i,:]-centers[i,...])*labels[:,tf.newaxis]
        loss += tf.reduce_mean(tf.square(diff))
    return loss


def get_divergence_loss(alpha,y_true,y_pred,centers):
    loss=0
    for i in range(AU_count):
        labels = tf.reshape(1-y_true[:,i],[-1])
        labels = tf.cast(labels,tf.float32)
        diff = alpha[i]*(y_pred[:,i,:]-centers[i,...])*labels[:,tf.newaxis]
        loss += tf.reduce_mean(tf.square(diff))
    return loss

def get_full_center_loss(alpha, feature_dim,ind):  
        # Build a graph
    graph = tf.Graph()
    with graph.as_default():
        centers = tf.get_variable('centers_{}'.format(ind), shape=[AU_count,256], 
                                  initializer=tf.constant_initializer(0),
                                  dtype=tf.float32)     # Create a variable tensor

    # Create a session, and run the graph
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()  # Initialize values of all variable tensors
        output_a = session.run(centers)            # Return the value of the variable tensor
   
    centers = output_a
    def center_loss(y_true, y_pred):        
#         a = np.zeros([AU_count, feature_dim])
        c = tf.constant(centers)
        c = get_centers(y_true,y_pred,c)
#         centers = c
#         centers.assign(c)
        num_loss = get_center_loss(alpha,y_true,y_pred,c)
        den_loss = get_divergence_loss(alpha,y_true,y_pred,c) 
        
        return num_loss/den_loss
    
    return center_loss


def weighted_bce(y_true, y_pred,weights):
    weights = ((y_true) * 59.) + 1.

    bce = K.binary_crossentropy(y_true, y_pred,from_logits=False)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce

def macro_soft_f1(weights):
    def loss(y, y_hat):
        """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
        Use probability values instead of binary predictions.

        Args:
            y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
            y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

        Returns:
            cost (scalar Tensor): value of the cost function for the batch
        """
        print(weights)
        y = tf.cast(y, tf.float32)
        y_hat = tf.cast(y_hat, tf.float32)
        tp = tf.reduce_sum(y_hat * y, axis=0)
        fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
        fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
        tn = tf.reduce_sum((1 - y_hat) * (1-y), axis=0)
        soft_f1 = (tp+ 1e-16) / (tp + 0.25*fn + 0.75*fp + 1e-16)
        
        cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
        macro_cost = tf.reduce_mean(cost*weights)+weighted_bce(y,y_hat,weights) # +average on all labels
        
        return macro_cost
    return loss 


def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    
        
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = (2*tp+1e-16) / (2*tp + fn + fp + 1e-16)
    
    macro_f1 = tf.reduce_mean(f1)#+tf.keras.losses.binary_crossentropy(y,y_hat)
    return macro_f1



def huber_loss(y_true, y_pred, clip_delta=0.1):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)
