import tensorflow as tf

def add_moments_summary(variable, scope, histogram=True):
    '''
    Attach tf.summaries to compute the mean, standard deviation, minimum, maximum, and,
    optionally, a histogram of a variable.

    Arguments:
        variable (TensorFlow Variable): A TensorFlow Variable of any shape to which to
            add the summary operations. Must be a numerical data type.
        scope (str): The name scope prefix for the summaries.
        histogram (bool, optional): If `True`, a histogram summary will be added.
    '''
    mean = tf.reduce_mean(variable)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(variable - mean)))
    variable_summaries = [tf.summary.scalar(name='{}_mean'.format(scope), tensor=mean),
                          tf.summary.scalar(name='{}_stddev'.format(scope), tensor=stddev),
                          tf.summary.scalar(name='{}_max'.format(scope), tensor=tf.reduce_max(variable)),
                          tf.summary.scalar(name='{}_min'.format(scope), tensor=tf.reduce_min(variable))]
    if histogram:
        variable_summaries.append(tf.summary.histogram(name='{}_histogram'.format(scope), tensor=variable))
    return tf.summary.merge(variable_summaries, name='{}_moments'.format(scope))

def add_mean_norm_summary(variable, scope, order='euclidean'):
    '''
    Attach a tf.summary to compute the l2 norm of the variable divided by its number
    of components.

    Arguments:
    variable (TensorFlow Variable): A TensorFlow Variable of any shape to which to
        add this summary operation. Must be a numerical data type.
    scope (str): The name scope prefix for the summaries.
    order (str, optional): The order of the norm. Supported values are 'euclidean', 1, 2, np.inf
        and any positive real number yielding the corresponding p-norm. Default is 'euclidean' which is
        equivalent to the Frobenius norm if variable is a matrix and equivalent to the 2-norm for vectors.
    '''
    mean_norm = tf.norm(variable, ord=order) / tf.to_float(tf.size(variable))
    return tf.summary.scalar('{}_mean_{}_norm'.format(scope, order), mean_norm)
