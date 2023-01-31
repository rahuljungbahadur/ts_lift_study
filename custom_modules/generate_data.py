import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from typing import List

class GenerateData:
    """
    Generates time series data for shoppers
    """
    def __init__(self, start_date:str, end_date:str, periodicity:List[str], freq:str = 'D') -> None:

        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.date_series = pd.date_range(start = start_date, end=end_date, freq=freq)
        self.periodicity = periodicity
        self.freq = freq
        days_in_yr = 365.2422     
        self.seasonal_dict = {'weekly':(days_in_yr/52), 'monthly':(days_in_yr/12), 'yearly':(days_in_yr)}

    def add_seasonality(self, periodicity):
        """
        Function that creates a sinusoidal time series

        Parameters
        ------------
        start_date (datetime):
            The start_date for the series
        end_date (datetime):
            The end date for the series
        periodicity (float):
            The desired cycle length
        freq (str) = 'D':
            The freq for generating the datetime series

        Returns
        ------------
        A pd.series.Series object with a series of specified periodicity
        """
        series_len = len(self.date_series)
        x = np.linspace(0, series_len, series_len)
        y = np.sin(x* (2*np.pi/periodicity))

        date_sine = pd.Series(data=y, index = self.date_series)

        return date_sine

    def generate_sales_data(self, mean_sales:float, error_sd:float, trt_start_date:str = '2021-01-01', is_exposed = False, treatment_period:int = 30, trend:float = 0.0):
        """
        A function which generates an additive sales data model with the given mean, trend, variance and seasonality 
        within the specified period.
     
        Parameters
        -------------
        start_date (str):
            The start date for the sequence
        end_date (str):
            End date
        mean_sales (float):
        errosr_sd (float):
        trend (float) = 0.0:
            The total increase from the mean towards the end of the series
        freq (str) = 'D':
            The freq for creating the dataset
        """
        
        self.date_series = pd.date_range(start = self.start_date, end=self.end_date, freq=self.freq)
        trend = np.linspace(0, trend, len(self.date_series), dtype=np.float64)
     
        if trt_start_date:
            treatment_outcome_zeros = pd.DataFrame(pd.Series(np.zeros(len(self.date_series)), index=self.date_series), columns=['zeros'])
            treatment_outcome_ones = (pd.DataFrame(pd.Series(np.ones(treatment_period),
             index = pd.date_range(start = trt_start_date, end = pd.to_datetime(trt_start_date) + timedelta(days = treatment_period-1))), columns=['ones']))
            treatment_outcome = treatment_outcome_zeros.join(treatment_outcome_ones).fillna(0).assign(total_trt = lambda d: d.zeros + d.ones).drop(['zeros', 'ones'], axis=1).total_trt
     
        sales_cyclicity = np.zeros(len(self.date_series), dtype=np.float64)
     
        for item in self.periodicity:
            sales_cyclicity = np.sum([sales_cyclicity, self.add_seasonality(periodicity=self.seasonal_dict[item])], axis=0)
     
     
     
        output = mean_sales + trend + sales_cyclicity + np.random.normal(loc = 0, scale=error_sd, size=len(self.date_series))
        max_output = output.max()
        output = np.clip(a = output, a_min = 0.0, a_max = max_output)
     
        output_pd_series = pd.Series(output, index = self.date_series, dtype=float)
        if is_exposed:
            output_pd_series = output_pd_series + treatment_outcome
     
        return np.round(output_pd_series, 2)



    def generate_data(self, campaign_start_date = '2022-03-01', campaign_end_date = '2022-11-01', treatment_period:int = 60, trt_proportion:float = 0.1, total_obs:int = 1000):
        """
        Generates the time series data for the treatment and control group, given the proportion for each
        """
        assert pd.to_datetime(campaign_end_date) <= self.end_date, print("campaign_end_date should be less than or equal to the end_date")

        num_exposed = int(total_obs * trt_proportion)
        is_exposed = np.repeat(a = 1, repeats=num_exposed)
        is_not_exposed = np.repeat(a = 0, repeats=total_obs - num_exposed)
        all_folks = np.concatenate([is_exposed, is_not_exposed], axis=-1)

        ## For the treated folks add 2 dollars for the time they are being treated

        output_df = pd.DataFrame({
            'shopper_id':np.random.randint(low=10000, high=99999, size=total_obs).astype(str),
            'shopper_mean':np.round(np.random.random(size=total_obs)*10, 2),
            'is_exposed':all_folks
        })

        output_df = (output_df.assign(campaign_start_date = pd.to_datetime(campaign_start_date)))


        output_df = (output_df.assign(
            trt_start_date = lambda d: d.campaign_start_date.apply(lambda x : x + timedelta(np.random.randint(low = 0, high = treatment_period))))
            )

        get_sales = output_df.apply(lambda d: self.generate_sales_data(mean_sales=d.shopper_mean,
            trt_start_date= d.trt_start_date, is_exposed = bool(d.is_exposed),
            trend = 0.5, error_sd=2.0), axis=1)
        get_sales = get_sales.reset_index(drop=True)

        output_df = pd.concat([output_df, get_sales], axis=1)
        output_df = output_df.drop('campaign_start_date', axis=1).melt(id_vars=['shopper_id', 'is_exposed', 'trt_start_date', 'shopper_mean'], value_name='dollar_sales', var_name='date')



        return output_df


    def plot_mean_sales(self, df):
        """
        Plots the mean sales for the treatment and control group
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.lineplot(data=df, x='date', y='dollar_sales', hue='is_exposed', ax=ax)
        ax.set_title('Mean Sales')
        ax.set_ylabel('Dollar Sales')
        ax.set_xlabel('Date')
        plt.show()
     