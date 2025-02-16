import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.rolling import RollingOLS

def maximum_drawdown(returns, annual_factor = 12):
    """Calculate the maximum drawdown for each asset for returns."""
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    return drawdowns.min()


def summary_statistics_annualized(returns, annual_factor=12):
    """Calculate annualized summary statistics for returns."""
    stats = pd.DataFrame(index=returns.columns)
    stats['Mean'] = returns.mean() * annual_factor
    stats['Vol'] = returns.std() * np.sqrt(annual_factor)
    stats['Sharpe'] = stats['Mean'] / stats['Vol']
    stats['Min'] = returns.min()
    stats['Max'] = returns.max()
    stats['Skewness'] = returns.skew()
    stats['Kurtosis'] = returns.kurtosis()
    stats['VaR 5%'] = returns.quantile(0.05)
    stats['CVaR 5%'] = returns[returns <= returns.quantile(0.05)].mean()
    drawdown_stats = maximum_drawdown(returns)
    stats = stats.join(drawdown_stats)
    return stats


def performance_metrics_annualized(returns, annual_factor=12):
    """Calculate performance metrics for returns."""
    stats = summary_statistics_annualized(returns, annual_factor)
    return stats[['Mean', 'Vol', 'Sharpe', 'Min', 'Max', 'Max Drawdown']]


def maximum_drawdown(returns):
    """Calculate the maximum drawdown for each asset for returns,
    and max/min/recovery dates within the max drawdown period."""
    assert not returns.isnull().any().any()
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() # series
    end_date = drawdowns.idxmin()
    summary = pd.DataFrame({'Max Drawdown': max_drawdown, 'Bottom': end_date})
    for col in drawdowns.columns:
        peak_date = rolling_max.loc[:end_date[col], col].idxmax()
        try:
            recovery = drawdowns.loc[end_date[col]:, col] # after the bottom
            recovery_date = recovery[recovery >= 0].index[0]
        except IndexError:
            recovery_date = None
        duration = recovery_date - peak_date if recovery_date else None
        summary.loc[col, 'Peak'] = peak_date
        summary.loc[col, 'Recovery'] = recovery_date
        summary.loc[col, 'Duration'] = duration
    return summary


def tail_metrics(returns):
    """Calculate tail risk metrics for returns."""
    stats = summary_statistics_annualized(returns)
    return stats[['Skewness', 'Kurtosis', 'VaR 5%', 'CVaR 5%', 
                  'Max Drawdown', 'Bottom', 'Peak',	'Recovery', 'Duration']]


def tangency_weights(returns, shrinkage=1):
    """Compute tangency portfolio weights with optional shrinkage."""
    cov = returns.cov()
    cov_shrunk = shrinkage * cov + (1 - shrinkage) * np.diag(np.diag(cov)) # 2D diagonal matrix
    cov_inv = np.linalg.inv(cov_shrunk * 12)  # Annualized covariance matrix
    means = returns.mean() * 12
    weights = cov_inv @ means / (np.ones(len(means)) @ cov_inv @ means)
    return pd.DataFrame(weights, index=returns.columns, columns=['Tangency Weights'])


def gmv_weights(returns, shrinkage=1):
    """Compute global minimum variance portfolio weights with optional shrinkage."""
    cov = returns.cov()
    cov_shrunk = shrinkage * cov + (1 - shrinkage) * np.diag(np.diag(cov))
    cov_inv = np.linalg.inv(cov_shrunk * 12)
    ones = np.ones(returns.shape[1])
    weights = cov_inv @ ones / (ones @ cov_inv @ ones)
    return pd.DataFrame(weights, index=returns.columns, columns=['GMV Weights'])


def mv_portfolio(returns, target_mean=0.0015):
    """Compute weights for tangency, GMV, and target MV portfolios."""
    mu_tan = returns.mean() @ tangency_weights(returns)['Tangency Weights']
    mu_gmv = returns.mean() @ gmv_weights(returns)['GMV Weights']
    delta = (target_mean - mu_gmv) / (mu_tan - mu_gmv)
    mv_weights = delta * tangency_weights(returns).values + (1 - delta) * gmv_weights(returns).values
    weights = pd.DataFrame(mv_weights, index=returns.columns, columns=['MV Weights'])
    weights['Tangency Weights'] = tangency_weights(returns).values
    weights['GMV Weights'] = gmv_weights(returns).values
    return weights


def mv_frontier(returns, delta_range=(-2, 2), grid_points=150):
    """Generate data for the mean-variance frontier."""
    w_tan = tangency_weights(returns, shrinkage=1)
    w_gmv = gmv_weights(returns, shrinkage=1)

    delta_grid = np.linspace(delta_range[0], delta_range[1], grid_points)
    mv_frame = pd.DataFrame(index=delta_grid, columns=['mean', 'vol'])

    for delta in delta_grid:
        omega_mv = delta * w_tan.values + (1 - delta) * w_gmv.values
        port_ret = returns @ omega_mv
        mv_frame.loc[delta, 'mean'] = port_ret.mean().values[0] * 12  # Annualized mean return
        mv_frame.loc[delta, 'vol'] = port_ret.std().values[0] * np.sqrt(12)  # Annualized volatility

    return mv_frame


def rolling_oos(returns, year_start=2013, year_end=2023, target_mean=0.015, tip_adj=0.0012):
    """Calculate rolling OOS portfolio performance across multiple years."""
    returns = returns.set_index('Date')
    port_oos = pd.DataFrame(index=returns.index, columns=[
        'Tangency', 'Tangency no TIP', 'Tangency hi TIP', 
        'Equally Weighted', 'Risk Parity', 'Regularized'], dtype=float)

    for year in range(year_start, year_end):
        returns_is = returns.loc[:str(year)]
        returns_oos = returns.loc[str(year + 1):]
        returns_hi_tip = returns_is.copy()
        returns_hi_tip['TIP'] += tip_adj

        weights_is = pd.DataFrame(index=returns_is.columns)
        weights_is['Tangency'] = tangency_weights(returns_is)
        weights_is['Tangency no TIP'] = tangency_weights(returns_is.drop(columns=['TIP']), shrinkage=1).reindex(returns_is.columns, fill_value=0)
        weights_is['Tangency hi TIP'] = tangency_weights(returns_hi_tip)
        weights_is['Equally Weighted'] = 1 / len(returns_is.columns)
        weights_is['Risk Parity'] = 1 / returns_is.var()
        weights_is['Regularized'] = tangency_weights(returns_is, shrinkage=0.5)

        weights_is *= target_mean / (returns_is.mean() @ weights_is)
        port_oos.loc[returns_oos.index] = returns_oos @ weights_is

    return port_oos.dropna()


def correlation_heatmap(returns):
    """Plot a heatmap of the correlation matrix of the returns, off-diagonal only.
    Print MIN, MAX of correlations."""
    corr_matrix = returns.corr()
    mask = np.eye(corr_matrix.shape[0], dtype=bool) # Highlight diagonal
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=mask, fmt='.2f')
    plt.xticks(rotation=45)
    plt.show()
    corrs = corr_matrix.unstack().sort_values(ascending=False)
    corrs = corrs[corrs != 1] # Remove self-correlations
    print(f"Max corr {max(corrs):.2f}: {corrs.idxmax()}")
    print(f"Min corr {min(corrs):.2f}: {corrs.idxmin()}")


def cross_sectional_regression(mean_returns, betas, intercept=False):
    """Calculate cross-section regression statistics for mean returns against respective betas."""
    if intercept:
        betas = sm.add_constant(betas)
    regr = sm.OLS(mean_returns, betas).fit()
    results = regr.params.to_frame('Cross-Sectional Regression')
    results.loc['R^2'] = regr.rsquared
    results.loc['MAE'] = regr.resid.abs().mean()
    return results


def time_series_regression_annualized(returns, factors, intercept=True, annual_factor=12):
    """Calculate regression statistics for each asset against factors (dataframe) or benchmark (series), annualized."""
    if intercept:
        factors = sm.add_constant(factors)
    results = pd.DataFrame(index=returns.columns)
    regrs = []
    for col in returns.columns:
        regr = sm.OLS(returns[col], factors, missing='drop').fit()
        regrs.append(regr)
        if intercept:
            results.loc[col, 'Alpha'] = regr.params[0] * annual_factor
            for i, factor in enumerate(factors.columns[1:]):
                results.loc[col, f'{factor} Beta'] = regr.params[i + 1]
            results.loc[col, 'Treynor Ratio'] = returns[col].mean() / sum(regr.params[1:]) * annual_factor
            results.loc[col, 'IR'] = regr.params[0] / regr.resid.std() * np.sqrt(annual_factor)
        else:
            for i, factor in enumerate(factors.columns):
                results.loc[col, f'{factor} Beta'] = regr.params[i]
            results.loc[col, 'Treynor Ratio'] = returns[col].mean() / sum(regr.params) * annual_factor
        results.loc[col, 'R^2'] = regr.rsquared
    return {'df': results, 'regrs': regrs}


def rolling_regression(returns, factors, window=60, intercept=True):
    """Calculate rolling regression statistics for each asset against factors, vectorized,
    and compare replication performance of static, is, oos against actual returns."""
    if intercept:
        factors = sm.add_constant(factors)
    regr = RollingOLS(returns, factors, window=window)
    rolling_betas = regr.fit().params
    replication_is = (rolling_betas * factors).sum(axis=1, skipna=False) # in-sample replication
    replication_oos = (rolling_betas.shift() * factors).sum(axis=1, skipna=False) # out-of-sample replication
    static_regr = sm.OLS(returns, factors).fit() # static regression as benchmark
    replication_static = static_regr.fittedvalues
    if intercept:
        return pd.DataFrame({'Actual': returns, 'Static IS YesInt': replication_static, 'Rolling IS YesInt': replication_is, 'Rolling OOS YesInt': replication_oos})
    return pd.DataFrame({'Actual': returns, 'Static IS NoInt': replication_static, 'Rolling IS NoInt': replication_is, 'Rolling OOS NoInt': replication_oos})
    

def expanding_regression(returns, factors, start_window=60, intercept=True):
    """Calculate out-of-sample R-squared using expanding window regression.
        returns (pd.Series): The dependent variable (e.g., asset returns).
        factors (pd.DataFrame): The independent variables (e.g., factors/signals."""
    factors = factors.shift()
    oos_errors = []
    null_errors = []
    if intercept:
        factors = sm.add_constant(factors)
    for t in range(start_window, len(returns)):
        regr = sm.OLS(returns.iloc[:t], factors.iloc[:t], missing='drop').fit()
        # oos_forecast = np.dot(regr.params, factors.iloc[t])
        oos_forecast = regr.predict(factors.iloc[t])[0]
        oos_errors.append(returns.iloc[t] - oos_forecast)
        null_forecast = returns.iloc[:t].mean()
        null_errors.append(returns.iloc[t] - null_forecast)
    oos_R2 = 1 - (np.array(oos_errors)**2).sum() / (np.array(null_errors)**2).sum()
    return oos_R2


def strategy_lagged_regression(returns, factors, intercept=True, annual_factor=12):
    """
    Calculate a strategy using lagged regression over the entire period 
    by assigning weights as 100 * prediction.
        returns (pd.DataFrame): Single-column DataFrame of dependent variable (e.g., asset returns).
        factors (pd.DataFrame): DataFrame of independent variables (e.g., factors/signals).
    Returns:
        pd.DataFrame: Strategy returns with weights based on lagged regression predictions.
    """
    factors = factors.shift()
    regr = time_series_regression_annualized(returns, factors, intercept, annual_factor)['regrs'][0]
    if intercept:
        factors = sm.add_constant(factors)
    # weights = 100 * (factors * regr.params).dropna().sum(axis=1)
    weights = 100 * regr.fittedvalues # prediction
    strategy_returns = (weights * returns.iloc[:, 0]).dropna()
    return pd.DataFrame(strategy_returns, columns=['-'.join(factors.columns)])


def strategy_expanding_regression(returns, factors, start_window=60, intercept=True):
    """Calculate strategy statistics using expanding regression."""
    factors = factors.shift()
    if intercept:
        factors = sm.add_constant(factors)
    oos_strategy_returns = []
    for t in range(start_window, len(returns)):
        regr = sm.OLS(returns.iloc[:t], factors.iloc[:t], missing='drop').fit()
        oos_forecast = regr.predict(factors.iloc[[t]])[0]
        weight = 100 * oos_forecast
        oos_strategy_returns.append(weight * returns.iloc[t, 0])
    oos_strategy_returns = pd.DataFrame(oos_strategy_returns, columns=['-'.join(factors.columns)], index=returns.index[start_window:])
    return oos_strategy_returns


def strategy_statistics_annualized(strategy_returns, market, risk_free_rate):
    """
    Calculate strategy statistics using lagged regression over the entire period.
        returns (pd.DataFrame): Single-column DataFrame of dependent variable (e.g., asset returns).
        factors (pd.DataFrame): DataFrame of independent variables (e.g., factors/signals).
        market (pd.DataFrame): DataFrame of market returns.
    Returns:
        pd.DataFrame: Strategy statistics.
    """
    sum_df = summary_statistics_annualized(strategy_returns)[['Mean', 'Vol', 'Sharpe', 'Max Drawdown', 'VaR 5%']]
    reg_df = time_series_regression_annualized(strategy_returns, market.loc[strategy_returns.index])['df'].add_prefix('Market ')
    negative_risk_premium_months = len(
        strategy_returns[strategy_returns.iloc[:, 0] - risk_free_rate.loc[strategy_returns.index].iloc[:, 0] < 0]
    )
    sum_df['Negative Risk Premium Months'] = negative_risk_premium_months
    return pd.concat([sum_df, reg_df], axis=1)