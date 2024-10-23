import numpy as np
from scipy import optimize

import matplotlib.pyplot as plt
from sklearn.utils.extmath import log_logistic
from scipy.special import expit

def quadratic_loss(u):
  return np.square(1-u)

def hinge_loss(u):
  l = 1-u
  return l.clip(0)

def logistic_loss(u):
  return np.log(1+np.exp(-u))

def solve_diagonal_ridge_closed_form(Theta, Samples, lambda_reg):
    n = (Theta[0]).shape[0]
    Pi_diag = np.zeros(n)
    for i in range(n):
        numerator = np.dot(Samples[:, i], Theta[:, i])
        denominator = np.dot(Theta[:, i], Theta[:, i]) + lambda_reg
        Pi_diag[i] = numerator / denominator
    #print("Pi diag", Pi_diag)

    return np.diag(Pi_diag)

def solve_diagonal_regularized(Theta, Samples, lambda_reg):
    n = Theta.shape[0]
    Pi_diag = np.zeros(n)
    for i in range(n):
        # Objective function with L2 regularization
        def objective(x):
            return np.sum((Samples[i, :] - x * Theta[i, :]) ** 2) + lambda_reg * np.abs(x)
        
        res = minimize(objective, x0=0)
        Pi_diag[i] = res.x if res.success else 0
    
    return np.diag(Pi_diag)


def DPR_logistic(w, X, y):
    alpha, theta =  w[-1], w[:-1]
    return np.sum(logistic_loss(y * (X @ theta + alpha)))

class Learner:
    def __init__(self, step_size=.1, Pi=None, Pi_learning=False):

        self.eval_history = []
        self.theta_history = []
        self.step_size = step_size
        self.samples_history = []
        self.Pi_history = []
        self.Pi_learning = Pi_learning
        self.Pi = Pi
        
        

    def init_model(self, X, y):
        # Initialize alpha and theta to zeros based on the features X
        self.alpha = 0
        self.theta = np.zeros(X.shape[1])
        self.mu = np.mean(X[y==-1] , axis=0) - np.mean(X[y==1], axis=0)

    def grad(self, X, y):
        # This method should be implemented by subclasses to compute gradients
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate(self, X, y):
        self.theta_history.append(self.theta.copy())

        predictions = self.alpha + np.dot(X, self.theta)
        predicted_labels = np.sign(predictions)
        predicted_labels[predicted_labels == 0] = -1  # Adjust zero predictions to -1 if necessary
        accuracy = np.mean(predicted_labels == y)
        self.eval_history.append(accuracy)

    def history(self):
        # Return the evaluation history and all the theta values across time
        self.theta_history = np.array(self.theta_history)
        return self.eval_history, self.theta_history

    def Pis_history(self):
        self.Pi_history = np.array(self.Pi_history)
        return self.Pi_history

    def get_theta(self):
        return self.theta

    def plot_sep(self, color='b'):
        x_values = np.linspace(-1, 2, 400)
        y = (-self.theta[0] / self.theta[1]) * x_values - (self.alpha / self.theta[1])
        plt.plot(x_values, y, color= color)

    def learn_Pi(self, X, y):
        if self.Pi_learning:
            # for learners that need to learn Pi
            self.samples_history.append(X[y==-1].mean(axis=0) - self.mu)
            if True:
                self.Pi = solve_diagonal_ridge_closed_form(np.array(self.theta_history), np.array(self.samples_history), 0)
                #print("this is pi", self.Pi)
            self.Pi_history.append(self.Pi)

    def plot_pi_convergence(self, pi_perf, linestyle='-', color='blue', legend=""):
        norms = [np.linalg.norm(pi - pi_perf, 'fro') for pi in self.Pi_history]
        plt.plot(norms, ls=linestyle, color=color, label=legend)
        #plt.title('Convergence of Pi Matrices')
        plt.xlabel('Iteration')
        plt.ylabel('Frobenius Norm of Difference')


class RRM(Learner):
    def grad(self, X, y):
        # Compute the logistic regression gradient
        m = optimize.minimize(DPR_logistic, np.append(self.alpha, self.theta), args=(X, y),method='BFGS')
        self.theta = m.x[:-1]
        self.alpha = m.x[-1]


class ARRM(Learner):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.RRM_step = True

    def grad(self, X, y):
        if self.RRM_step:
            # Compute the logistic regression gradient
            m = optimize.minimize(DPR_logistic, np.append(self.alpha, self.theta), args=(X, y),method='BFGS')
            self.theta = m.x[:-1]
            self.alpha = m.x[-1]
            self.RRM_step = False
        else:
            # optimize the distribution with respect to the current parameters is blatantly stupid here
            n = self.theta.shape[0]
            self.theta += np.ones(n)
            self.RRM_step = True


class RGD(Learner):
    def grad(self, X, y):
        n = X.shape[0]
        # compute the gradient of non-performative risk
        yz =  y * (np.dot(X, self.theta) + self.alpha)
        z = expit(yz)
        z0 = (z - 1) * y

        self.alpha -= self.step_size * z0.mean()
        self.theta -= self.step_size * X.T @ z0 / n

class PiRGD(Learner):
    # TODO explain why this regalarization is good...
    def __init__(self, lamb=.01,*args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.lamb = lamb

    def grad(self, X, y):
        n = X.shape[0]
        # Compute the gradient of the logistic loss
        yz = y * (np.dot(X, self.theta) + self.alpha)
        z = expit(yz)  
        z0 = (z - 1) * y
        
        # Update alpha and theta with the inclusion of the Pi matrix term in theta update
        self.alpha -= self.step_size * np.mean(z0)
        self.theta -= self.step_size * (np.dot(X.T, z0) / n + self.lamb *self.theta)



class RPPerfGD(Learner):
    def __init__(self, lamb=.01,*args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.lamb = lamb
    # TODO explain why this regalarization is good...
    def grad(self, X, y):
        n, n1 = X.shape[0], X[y==-1].shape[0]
        yz = y * (np.dot(X, self.theta) + self.alpha)
        z = expit(yz) # 1/(1+np.exp(-x))
        z0 = (z - 1) * y
        
        self.theta -= self.step_size * self.Pi @ self.theta * z0[y==-1].sum()/n1

        self.theta -= self.step_size * (np.dot(X.T, z0) / n + self.lamb *self.theta)

        self.alpha -= self.step_size * np.mean(z0)


class PiRGD2(Learner):
    # TODO explain why this regalarization is good...
    def __init__(self,  lamb=.01,*args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.lamb = lamb

    def grad(self, X, y):
        n = X.shape[0]
        # Compute the gradient of the logistic loss
        z = np.dot(X, self.theta) + self.alpha
        z0 = z - y
        
        # Update alpha and theta with the inclusion of the Pi matrix term in theta update
        self.alpha -= self.step_size * np.mean(z0)
        self.theta -= self.step_size * (np.dot(X.T, z0) / n + self.lamb *self.theta )#np.dot(self.Pi, self.theta))


class RPPerfGD2(Learner):
    def grad(self, X, y):
        n, n1 = X.shape[0], X[y==-1].shape[0]
        z = np.dot(X, self.theta) + self.alpha
        z0 = z - y
        
        # Update alpha and theta with the inclusion of the Pi matrix term in theta update
        self.alpha -= self.step_size * np.mean(z0)
        self.theta -= self.step_size * (np.dot(X.T, z0) / n )#np.dot(self.Pi, self.theta))
        
        #self.theta -= self.step_size * np.square(z0[y==-1]).mean() * self.Pi.T @ (X[y==-1].mean(axis=0) - self.Pi @ self.theta)
        self.theta -= self.step_size * self.Pi @ self.theta * z0[y==-1].sum()/n1 # todo check sign



class SFPerfGD2(Learner):
    def __init__(self, sigma=.01, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.sigma = sigma
    def grad(self, X, y):
        n, d = X.shape[0], X.shape[1]
        n1 = X[y==-1].shape[0]
        z = np.dot(X, self.theta) + self.alpha
        z0 = z - y
        

        
        # Update alpha and theta with the inclusion of the Pi matrix term in theta update
        self.alpha -= self.step_size * np.mean(z0)

        update1 = self.step_size * (np.dot(X.T, z0) / n )
        if not np.isnan(update1).any():
            self.theta -= update1
        u = X[y==-1] - self.mu - self.Pi @ self.theta
        su = np.square(z0[y==-1]).reshape((1,n1))
        
        update = self.step_size * self.Pi @ (np.dot(su, u).reshape(d))/(2*self.sigma**2 *n1)
        if not np.isnan(update).any():
            self.theta -= update



class SFPerfGD(Learner):
    def __init__(self, sigma=.01, lamb=1, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        self.sigma = sigma
        self.lamb = lamb
    def grad(self, X, y):
        n, d = X.shape[0], X.shape[1]
        n1 = X[y==-1].shape[0]

        # compute the gradient of non-performative risk
        yz =  y * (np.dot(X, self.theta) + self.alpha)
        z = expit(yz)
        z0 = (z - 1) * y
        
        self.alpha -= self.step_size * np.mean(z0)
        self.theta -= self.step_size * (np.dot(X.T, z0) / n  + self.lamb * self.theta)
        u = X[y==-1] - self.mu - self.Pi @ self.theta
        try:
            su = -log_logistic(yz)[y==-1].reshape((1,n1))
            self.theta -= self.step_size * self.Pi @ (np.dot(su, u).reshape(d))/(2*self.sigma**2 *n1)
        except ValueError as e:
            print("Error:", e)
            # Handle the error, for example, by setting the update to zero or some other fallback
            self.theta -= np.zeros_like(self.theta)

        