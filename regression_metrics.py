# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:23:35 2016

@author: rwester

Utility script containing metrics for regression validation

"""
import numpy as np
from sklearn.linear_model import LinearRegression

class forecast_metrics(object):
    def __init__(self, y_true, y_pred, verbose=False):
        self.y_true = y_true
        self.y_pred = y_pred
        self.verbose = verbose
        self.result_store = {} # init as blank dict
        self.acc_ratio_arry = None # init as None
        self.linear_fit = None

    def __clean_zeros(self):
        mask = self.y_true != 0
        y_true_clean = self.y_true[mask]
        y_pred_clean = self.y_pred[mask]
        records_lost = self.y_true.shape[0] - y_true_clean.shape[0]
        if records_lost > 0 and self.verbose == True:
            print "Warning: ignoring "+str(records_lost)+" records due to non-zero constaint"
        return y_true_clean, y_pred_clean

    def __calc_acc_ratio_arry(self):
        # calculates the absolute accuracy ratio as an arry, is problematic if zeros
        # are found in denominator so drops all zeros and corresponding values in y_pred        
        y_true_nozeros, y_pred_nozeros = self.__clean_zeros()
        self.acc_ratio_arry = np.abs(y_pred_nozeros / y_true_nozeros)
        
    def __check_binary(self, array, return_mask=False):
        isbinary_mask = np.in1d(self.y_true, [0, 1])
        if return_mask == True:
            # returns mask where False is error values
            return isbinary_mask
        else:
            if np.all(isbinary_mask) == False:
                raise ValueError("Array contains values not equal to 0 or 1!")
            
    def __check_range(self, array, lower_bound, upper_bound, return_mask=False):
        # inclusive of lower and upper bounds
        inrange_mask = np.logical_or(array < lower_bound, array > upper_bound)
        if return_mask == True:
            # returns mask where False is error values
            return ~inrange_mask
        else:
            if np.any(inrange_mask) == True:
                raise ValueError("Array contains values out of range!")

    def calc_mse(self, return_result=False):
        # calculates the mean squared error
        self.result_store['mse'] = np.mean((self.y_pred-self.y_true)**2)
        if return_result == True:
            return self.result_store['mse']
    
    def calc_rmse(self, return_result=False):
        # calculates the root mean squared error
        self.result_store['rmse'] = np.sqrt(np.mean((self.y_pred-self.y_true)**2))
        if return_result == True:
            return self.result_store['rmse']
            
    def calc_mae(self, return_result=False):
        # calculates the mean absolute error
        self.result_store['mae'] = np.mean(np.abs(self.y_true - self.y_pred))
        if return_result == True:
            return self.result_store['mae']
                    
    def calc_mape(self, return_result=False):
        # calculates the mean absolute percentage error, is problematic if zeros are found
        # in denominator so drops all zeros and corresponding values in y_pred
        
        # filter out all zeros in denominator
        y_true_nozeros, y_pred_nozeros = self.__clean_zeros()
        
        # calculate mean absolute percentage error      
        self.result_store['mape'] = np.mean(np.abs((y_true_nozeros - y_pred_nozeros) / y_true_nozeros)) * 100      
        if return_result == True:
            return self.result_store['mape']
            
    def calc_acc_ratio(self, return_result=False):
        # if accuracy ratio array doesnt exist, then calculate it
        if not isinstance(self.acc_ratio_arry, np.ndarray):
            self.__calc_acc_ratio_arry()
            
        # calculate mean accuracy ratio
        self.result_store['acc_ratio'] = np.nanmean(self.acc_ratio_arry)
        if return_result == True:
            return self.result_store['acc_ratio']

    def calc_log_acc_ratio(self, return_result=False):        
        # if accuracy ratio array doesnt exist, then calculate it
        if not isinstance(self.acc_ratio_arry, np.ndarray):
            self.__calc_acc_ratio_arry()

        # calculate mean log accuracy ratio
        log_acc_ratio = np.log(self.acc_ratio_arry)
        self.result_store['log_acc_ratio'] = np.nanmean(log_acc_ratio[~np.isinf(log_acc_ratio)]) # take mean ignoring nan and inf
        if return_result == True:
            return self.result_store['log_acc_ratio']
          
    def calc_total_sos(self, return_result=False):
        # calculates the total sum of squares
        self.result_store['total_ss'] = np.sum((self.y_true-np.mean(self.y_true))**2)
        if return_result == True:
            return self.result_store['total_ss']
            
    def calc_reg_sos(self, return_result=False):
        # calculates the regression sum of squares, aka explained sum of squares        
        self.result_store['reg_ss'] = np.sum((self.y_pred-np.mean(self.y_true))**2)
        if return_result == True:
            return self.result_store['reg_ss']
            
    def calc_res_sos(self, return_result=False):
        # calculates the sum of squares of residuals, aka residual sum of squares   
        self.result_store['res_ss'] = np.sum((self.y_true-self.y_pred)**2)
        if return_result == True:
            return self.result_store['res_ss']
            
    def calc_r2(self, return_result=False):
        # calculates the R2 as 1 - SSres/SStotal
        self.result_store['r2'] = 1 - (self.calc_res_sos(return_result=True) / self.calc_total_sos(return_result=True))
        if return_result == True:
            return self.result_store['r2']
        
    def calc_fitted_r2(self, return_result=False):
        # Fits data with linear reg and calcs r2
        if not isinstance(self.linear_fit, LinearRegression):
            self.y_true_reshaped = self.y_true.reshape(-1, 1)
            self.y_pred_reshaped = self.y_pred.reshape(-1, 1)
            self.linear_fit = LinearRegression()
            self.linear_fit.fit(self.y_true_reshaped, self.y_pred_reshaped)
        self.result_store['fitted_r2'] = self.linear_fit.score(self.y_true_reshaped, self.y_pred_reshaped)
        if return_result == True:
            return self.result_store['fitted_r2']
            
    def calc_fitted_coef(self, return_result=False):       
        if not isinstance(self.linear_fit, LinearRegression):
            self.y_true_reshaped = self.y_true.reshape(-1, 1)
            self.y_pred_reshaped = self.y_pred.reshape(-1, 1)
            self.linear_fit = LinearRegression()
            self.linear_fit.fit(self.y_true_reshaped, self.y_pred_reshaped)
        self.result_store['fitted_coef'] = self.linear_fit.coef_[0][0]
        if return_result == True:
            return self.result_store['fitted_coef']
            
    def calc_brier_score(self, return_result=False):
        # calculates the brier score where y_true must be 1 or 0 (1 if the event happened,
        # 0 if not), y_pred is the probability from 0 to 1
        try:
            self.__check_binary(self.y_true)
            self.__check_range(self.y_pred, 0, 1)
            self.result_store['brier_score'] = np.mean((self.y_pred-self.y_true)**2)
        except ValueError as e:
            print "Error in Brier Score: " + str(e) + " (returning nan)"
            self.result_store['brier_score'] = np.nan
        if return_result == True:
            return self.result_store['brier_score']
                       
    def calc_all(self, return_result=False):
        # Calculates all metrics but brier score
        self.calc_mse()
        self.calc_rmse()
        self.calc_mae()
        self.calc_mape()
        self.calc_acc_ratio()
        self.calc_log_acc_ratio()
        self.calc_r2()
        self.calc_reg_sos()
        self.calc_fitted_r2()
        self.calc_fitted_r2()
        if return_result == True:
            return self.result_store
        
    def return_results(self):
        return self.result_store
