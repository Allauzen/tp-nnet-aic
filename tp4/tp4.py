#############################
## Regularization
#############################
def dropout(p,h,hp=None):
    """
        Perform the dropout transformation to the activation values

        :param p: the probability of dropout
        :param h: the activation values
        :param hp: the derivatives w.r.t. the pre-activation values
        :type p: float
        :type h: ndarray
        :type hp: ndarray
        :return mask: the bernoulli mask
        :return h: the transformed activation values
        :return hp: the transformed derivatives w.r.t. z
        :rtype h: ndarray
        :rtype hp: ndarray
    """


    return mask,h,hp



def updateParams(theta, dtheta, eta, regularizer=None, my_lambda=0.):
    """
        Perform the update of the parameters with the 
        possibility to do L1 or L2 regularization 

        :param theta: the network parameters
        :param dtheta: the updates of the parameters
        :param eta: the step-size of the gradient descent 
        :param regularizer: the name of the regularizer
        :param my_lambda: the value of the regularizer
        :type theta: ndarray
        :type dtheta: ndarray
        :type eta: float
        :type regularizer: str
        :type my_lambda: float
        :return: the parameters updated 
        :rtype: ndarray
    """

    if regularizer==None:
        return theta - eta * dtheta
