# 导入所需库
import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dask.array as da
import matplotlib as mpl
from scipy.stats import norm, shapiro
import pymannkendall as mk
import pkg_resources

# 检查 pymannkendall 版本
try:
    mk_version = pkg_resources.get_distribution("pymannkendall").version
    print(f"pymannkendall version: {mk_version}")
except pkg_resources.DistributionNotFound:
    print("pymannkendall is not installed. Please install it using: pip install pymannkendall")
    raise

# 手动实现 Pettitt 检验
def pettitt_test_manual(data):
    n = len(data)
    if n < 2:
        return False, 0, 1.0
    U_t = np.zeros(n)
    for t in range(n):
        for i in range(n):
            U_t[t] += np.sign(data[t] - data[i])
    K = np.max(np.abs(U_t))
    cp = np.argmax(np.abs(U_t))
    p = 2 * np.exp(-6 * K**2 / (n**3 + n**2))
    h = p < 0.05
    return h, cp, p

# 设置全局绘图样式
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['legend.title_fontsize'] = 14
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2

# 设置文件路径
output_dir = "D:/BaiduNetdiskDownload/figures"
nc_mean_path = os.path.join(output_dir, "ndvi_mean_1985_2022.nc")
nc_yearly_path = os.path.join(output_dir, "ndvi_yearly_1985_2022.nc")
basin_shp_path = "E:/iCloudDrive/Desktop/博士期间/毕业论文/data/30子流域.shp"
LAKES_SHP = r'E:\iCloudDrive\Desktop\博士期间\论文\蓝藻水华\2023.12.01\data\sum_lake.shp'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 检查文件是否存在
for path in [basin_shp_path, LAKES_SHP, nc_mean_path, nc_yearly_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

# 读取 shapefile 并转换为 WGS84 坐标系
basin_shp = gpd.read_file(basin_shp_path).to_crs("EPSG:4326")
lakes_shp = gpd.read_file(LAKES_SHP).to_crs("EPSG:4326")

# 获取流域边界框并添加缓冲区
bounds = basin_shp.total_bounds
buffer = 0.1
minx, miny, maxx, maxy = bounds[0] - buffer, bounds[1] - buffer, bounds[2] + buffer, bounds[3] + buffer
print(f"Bounds with buffer: [{minx}, {miny}, {maxx}, {maxy}]")

# 加载 NetCDF 文件
try:
    ds_mean = xr.open_dataset(nc_mean_path, chunks={'lat': 100, 'lon': 100})
    ds_yearly = xr.open_dataset(nc_yearly_path, chunks={'year': 10, 'lat': 100, 'lon': 100})
except Exception as e:
    raise ValueError(f"Error loading NetCDF files: {e}")

# 设置空间维度和 CRS
try:
    ndvi_mean = ds_mean['NDVI'].rename({'lat': 'y', 'lon': 'x'})
    ndvi_yearly = ds_yearly['NDVI'].rename({'lat': 'y', 'lon': 'x'})
    ndvi_mean = ndvi_mean.rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs("EPSG:4326")
    ndvi_yearly = ndvi_yearly.rio.set_spatial_dims(x_dim='x', y_dim='y').rio.write_crs("EPSG:4326")
except Exception as e:
    raise ValueError(f"Error setting spatial dimensions: {e}")

# 裁剪 NDVI 数据到流域范围
ndvi_mean = ndvi_mean.sel({'x': slice(minx, maxx), 'y': slice(maxy, miny)})
ndvi_yearly = ndvi_yearly.sel({'x': slice(minx, maxx), 'y': slice(maxy, miny)})

# 使用流域 shapefile 进行掩膜
ndvi_mean = ndvi_mean.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)
ndvi_yearly = ndvi_yearly.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)

# 移除 Inf 和负值
ndvi_mean = ndvi_mean.where(~da.isinf(ndvi_mean) & (ndvi_mean >= 0), np.nan)
ndvi_yearly = ndvi_yearly.where(~da.isinf(ndvi_yearly) & (ndvi_yearly >= 0), np.nan)

# --- 1. 逐年计算 NDVI 统计信息 ---
all_years = ds_yearly['year'].values
yearly_stats = []

for year in all_years:
    ndvi_year = ndvi_yearly.sel(year=year).compute()
    mean_ndvi = ndvi_year.mean(skipna=True).values.item()
    min_ndvi = ndvi_year.min(skipna=True).values.item()
    max_ndvi = ndvi_year.max(skipna=True).values.item()
    std_ndvi = ndvi_year.std(skipna=True).values.item()
    
    mean_ndvi = 0 if np.isnan(mean_ndvi) or np.isinf(mean_ndvi) else mean_ndvi
    min_ndvi = 0 if np.isnan(min_ndvi) or np.isinf(min_ndvi) else min_ndvi
    max_ndvi = 0 if np.isnan(max_ndvi) or np.isinf(max_ndvi) else max_ndvi
    std_ndvi = 0 if np.isnan(std_ndvi) or np.isinf(std_ndvi) else std_ndvi
    
    cv_ndvi = std_ndvi / mean_ndvi if mean_ndvi != 0 else 0
    
    yearly_stats.append({
        'Year': year,
        'Mean_NDVI': mean_ndvi,
        'Min_NDVI': min_ndvi,
        'Max_NDVI': max_ndvi,
        'Std_NDVI': std_ndvi,
        'CV_NDVI': cv_ndvi
    })

yearly_stats_df = pd.DataFrame(yearly_stats)

# --- 2. 选择年份 ---
target_years = [1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
selected_years = [year for year in target_years if year in all_years]
if 2022 not in selected_years:
    selected_years.append(2022)

print(f"Selected years for distribution plots: {selected_years}")

# --- 3. 计算 Sen 斜率和 MK 检验 ---
def sen_slope_and_mk(data):
    if np.all(np.isnan(data)) or len(data) < 2:
        return np.nan, np.nan, np.nan
    result = mk.original_test(data)
    return result.slope, result.p, 1 if result.trend == 'increasing' else -1 if result.trend == 'decreasing' else 0

def apply_sen_slope_and_mk(da):
    da = da.chunk({'year': -1})
    result = xr.apply_ufunc(
        sen_slope_and_mk,
        da,
        input_core_dims=[['year']],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float, float]
    )
    return result

sen_slope, p_value, trend_direction = apply_sen_slope_and_mk(ndvi_yearly)

sen_slope_da = xr.DataArray(sen_slope, coords={'y': ndvi_yearly['y'], 'x': ndvi_yearly['x']}, dims=['y', 'x']).rio.write_crs("EPSG:4326")
p_values_da = xr.DataArray(p_value, coords={'y': ndvi_yearly['y'], 'x': ndvi_yearly['x']}, dims=['y', 'x']).rio.write_crs("EPSG:4326")
trend_directions_da = xr.DataArray(trend_direction, coords={'y': ndvi_yearly['y'], 'x': ndvi_yearly['x']}, dims=['y', 'x']).rio.write_crs("EPSG:4326")

sen_slope_da = sen_slope_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)
p_values_da = p_values_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)
trend_directions_da = trend_directions_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)

sen_slope_mean = sen_slope_da.mean(skipna=True).values.item()
sen_slope_min = sen_slope_da.min(skipna=True).values.item()
sen_slope_max = sen_slope_da.max(skipna=True).values.item()
sen_slope_std = sen_slope_da.std(skipna=True).values.item()

# 计算显著性比例
total_pixels = np.prod(p_values_da.shape)
valid_pixels = total_pixels - np.isnan(p_values_da).sum().values.item()
significant_increase = (p_values_da < 0.05) & (trend_directions_da == 1)
significant_decrease = (p_values_da < 0.05) & (trend_directions_da == -1)
no_significant_change = (p_values_da >= 0.05)

increase_pixels = significant_increase.sum().values.item()
decrease_pixels = significant_decrease.sum().values.item()
no_change_pixels = no_significant_change.sum().values.item()

increase_ratio = increase_pixels / valid_pixels * 100 if valid_pixels > 0 else 0
decrease_ratio = decrease_pixels / valid_pixels * 100 if valid_pixels > 0 else 0
no_change_ratio = no_change_pixels / valid_pixels * 100 if valid_pixels > 0 else 0

# 计算显著性区域的 NDVI 变化量
ndvi_1985 = ndvi_yearly.sel(year=1985).compute()
ndvi_2022 = ndvi_yearly.sel(year=2022).compute()
ndvi_change = ndvi_2022 - ndvi_1985

ndvi_change_increase = ndvi_change.where(significant_increase)
ndvi_change_decrease = ndvi_change.where(significant_decrease)

ndvi_change_increase_mean = ndvi_change_increase.mean(skipna=True).values.item()
ndvi_change_increase_min = ndvi_change_increase.min(skipna=True).values.item()
ndvi_change_increase_max = ndvi_change_increase.max(skipna=True).values.item()

ndvi_change_decrease_mean = ndvi_change_decrease.mean(skipna=True).values.item()
ndvi_change_decrease_min = ndvi_change_decrease.min(skipna=True).values.item()
ndvi_change_decrease_max = ndvi_change_decrease.max(skipna=True).values.item()

# --- 4. 计算 NDVI 等级比例 ---
ndvi_levels = []
for year in selected_years:
    ndvi_year = ndvi_yearly.sel(year=year).compute()
    total_pixels = np.prod(ndvi_year.shape)
    low_pixels = (ndvi_year < 0.3).sum().values.item()
    medium_pixels = ((ndvi_year >= 0.3) & (ndvi_year <= 0.6)).sum().values.item()
    high_pixels = (ndvi_year > 0.6).sum().values.item()
    
    valid_pixels = total_pixels - np.isnan(ndvi_year).sum().values.item()
    low_ratio = low_pixels / valid_pixels * 100 if valid_pixels > 0 else 0
    medium_ratio = medium_pixels / valid_pixels * 100 if valid_pixels > 0 else 0
    high_ratio = high_pixels / valid_pixels * 100 if valid_pixels > 0 else 0
    
    ndvi_levels.append({
        'Year': year,
        'Low (<0.3) (%)': low_ratio,
        'Medium (0.3-0.6) (%)': medium_ratio,
        'High (>0.6) (%)': high_ratio
    })

ndvi_levels_df = pd.DataFrame(ndvi_levels)

# --- 5. 突变点检测（Pettitt 检验） ---
ndvi_yearly_mean = ndvi_yearly.mean(dim=['y', 'x'], skipna=True).compute()

if hasattr(mk, 'pettitt_test'):
    pettitt_result = mk.pettitt_test(ndvi_yearly_mean.values)
    change_point_year = all_years[pettitt_result.cp] if pettitt_result.h else None
    change_point_detected = pettitt_result.h
    change_point_p_value = pettitt_result.p
else:
    print("pymannkendall does not support pettitt_test. Using manual Pettitt test instead.")
    change_point_detected, cp, change_point_p_value = pettitt_test_manual(ndvi_yearly_mean.values)
    change_point_year = all_years[cp] if change_point_detected else None

# --- 6. 正态分布检验（Shapiro-Wilk 检验） ---
normality_tests = []
for year in selected_years:
    ndvi_year = ndvi_yearly.sel(year=year).compute()
    ndvi_values = ndvi_year.values.flatten()
    ndvi_values = ndvi_values[~np.isnan(ndvi_values)]
    if len(ndvi_values) > 5000:
        ndvi_values = np.random.choice(ndvi_values, 5000, replace=False)
    stat, p = shapiro(ndvi_values)
    normality_tests.append({
        'Year': year,
        'Shapiro-Wilk Stat': stat,
        'p-value': p,
        'Normal Distribution': 'Yes' if p > 0.05 else 'No'
    })

normality_df = pd.DataFrame(normality_tests)

# --- 7. 整合所有统计数据到一个汇总 CSV ---
summary_stats = yearly_stats_df.copy()

# 添加 NDVI 等级比例
summary_stats = summary_stats.merge(ndvi_levels_df, on='Year', how='left')

# 添加正态分布检验结果
summary_stats = summary_stats.merge(normality_df, on='Year', how='left')

# 添加显著性比例和 NDVI 变化量
summary_stats['Significant Increase (%)'] = np.nan
summary_stats['Significant Decrease (%)'] = np.nan
summary_stats['No Significant Change (%)'] = np.nan
summary_stats['NDVI Change Increase Mean'] = np.nan
summary_stats['NDVI Change Increase Range Min'] = np.nan
summary_stats['NDVI Change Increase Range Max'] = np.nan
summary_stats['NDVI Change Decrease Mean'] = np.nan
summary_stats['NDVI Change Decrease Range Min'] = np.nan
summary_stats['NDVI Change Decrease Range Max'] = np.nan

summary_stats.loc[summary_stats['Year'] == 1985, 'Significant Increase (%)'] = increase_ratio
summary_stats.loc[summary_stats['Year'] == 1985, 'Significant Decrease (%)'] = decrease_ratio
summary_stats.loc[summary_stats['Year'] == 1985, 'No Significant Change (%)'] = no_change_ratio
summary_stats.loc[summary_stats['Year'] == 1985, 'NDVI Change Increase Mean'] = ndvi_change_increase_mean
summary_stats.loc[summary_stats['Year'] == 1985, 'NDVI Change Increase Range Min'] = ndvi_change_increase_min
summary_stats.loc[summary_stats['Year'] == 1985, 'NDVI Change Increase Range Max'] = ndvi_change_increase_max
summary_stats.loc[summary_stats['Year'] == 1985, 'NDVI Change Decrease Mean'] = ndvi_change_decrease_mean
summary_stats.loc[summary_stats['Year'] == 1985, 'NDVI Change Decrease Range Min'] = ndvi_change_decrease_min
summary_stats.loc[summary_stats['Year'] == 1985, 'NDVI Change Decrease Range Max'] = ndvi_change_decrease_max

# 添加突变点检测结果
summary_stats['Change Point Detected'] = np.nan
summary_stats['Change Point Year'] = np.nan
summary_stats['Change Point p-value'] = np.nan
summary_stats.loc[summary_stats['Year'] == 1985, 'Change Point Detected'] = change_point_detected
summary_stats.loc[summary_stats['Year'] == 1985, 'Change Point Year'] = change_point_year
summary_stats.loc[summary_stats['Year'] == 1985, 'Change Point p-value'] = change_point_p_value

# 保存汇总统计数据
summary_stats.to_csv(os.path.join(output_dir, "ndvi_summary_stats.csv"), index=False)
print("Summary statistics saved to 'ndvi_summary_stats.csv'")

# --- 8. 绘制图表 ---
# 图表 1: NDVI 均值时间序列（使用 Sen+MK 方法）
plt.figure(figsize=(12, 6))
plt.plot(yearly_stats_df['Year'], yearly_stats_df['Mean_NDVI'], marker='o', color='black', linewidth=2.5, markersize=8, label='Mean NDVI')

# 使用 Sen+MK 计算整体趋势和分阶段趋势
# 1985-2022
result = mk.original_test(yearly_stats_df['Mean_NDVI'].values)
slope, p_value = result.slope, result.p
plt.plot(yearly_stats_df['Year'], yearly_stats_df['Mean_NDVI'].values[0] + slope * np.arange(len(yearly_stats_df)), color='red', linestyle='--', label=f'1985-2022: Slope={slope:.4f}{"**" if p_value < 0.01 else "*" if p_value < 0.05 else ""}')

# 1985-1990
subset = yearly_stats_df[(yearly_stats_df['Year'] >= 1985) & (yearly_stats_df['Year'] <= 1990)]
result = mk.original_test(subset['Mean_NDVI'].values)
slope, p_value = result.slope, result.p
plt.plot(subset['Year'], subset['Mean_NDVI'].values[0] + slope * np.arange(len(subset)), color='yellow', linestyle='--', label=f'1985-1990: Slope={slope:.4f}{"**" if p_value < 0.01 else "*" if p_value < 0.05 else ""}')

# 1991-2000
subset = yearly_stats_df[(yearly_stats_df['Year'] >= 1991) & (yearly_stats_df['Year'] <= 2000)]
result = mk.original_test(subset['Mean_NDVI'].values)
slope, p_value = result.slope, result.p
plt.plot(subset['Year'], subset['Mean_NDVI'].values[0] + slope * np.arange(len(subset)), color='green', linestyle='--', label=f'1991-2000: Slope={slope:.4f}{"**" if p_value < 0.01 else "*" if p_value < 0.05 else ""}')

# 2001-2010
subset = yearly_stats_df[(yearly_stats_df['Year'] >= 2001) & (yearly_stats_df['Year'] <= 2010)]
result = mk.original_test(subset['Mean_NDVI'].values)
slope, p_value = result.slope, result.p
plt.plot(subset['Year'], subset['Mean_NDVI'].values[0] + slope * np.arange(len(subset)), color='blue', linestyle='--', label=f'2001-2010: Slope={slope:.4f}{"**" if p_value < 0.01 else "*" if p_value < 0.05 else ""}')

# 2011-2022
subset = yearly_stats_df[(yearly_stats_df['Year'] >= 2011) & (yearly_stats_df['Year'] <= 2022)]
result = mk.original_test(subset['Mean_NDVI'].values)
slope, p_value = result.slope, result.p
plt.plot(subset['Year'], subset['Mean_NDVI'].values[0] + slope * np.arange(len(subset)), color='purple', linestyle='--', label=f'2011-2022: Slope={slope:.4f}{"**" if p_value < 0.01 else "*" if p_value < 0.05 else ""}')

plt.xlabel('Year')
plt.ylabel('NDVI')
plt.title('Figure 1: NDVI Mean with Trends (1985-2022, Excluding Lakes)', pad=15)
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "figure1_ndvi_mean_trends.pdf"), dpi=600, bbox_inches='tight')
plt.close()

# 图表 2: NDVI 等级比例
plt.figure(figsize=(12, 6))
plt.stackplot(ndvi_levels_df['Year'], 
              ndvi_levels_df['Low (<0.3) (%)'], 
              ndvi_levels_df['Medium (0.3-0.6) (%)'], 
              ndvi_levels_df['High (>0.6) (%)'], 
              labels=['Low (<0.3)', 'Medium (0.3-0.6)', 'High (>0.6)'],
              colors=sns.color_palette("Set2", 3),
              alpha=0.8)
plt.xlabel('Year')
plt.ylabel('Area Ratio (%)')
plt.title('Figure 2: NDVI Levels Ratio (1985-2022, Excluding Lakes)', pad=15)
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "figure2_ndvi_levels_ratio.pdf"), dpi=600, bbox_inches='tight')
plt.close()

# 图表 3: NDVI 空间异质性
plt.figure(figsize=(12, 6))
plt.plot(yearly_stats_df['Year'], yearly_stats_df['CV_NDVI'], marker='o', color='#1f77b4', linewidth=2.5, markersize=8, label='Coefficient of Variation')

# 标注关键年份的变异系数
key_years = [1985, 1990, 2000, 2010, 2015, 2020, 2022]
for year in key_years:
    cv_value = yearly_stats_df[yearly_stats_df['Year'] == year]['CV_NDVI'].values[0]
    plt.text(year, cv_value + 0.005, f'{cv_value:.3f}', ha='center', va='bottom', fontsize=12)

plt.xlabel('Year')
plt.ylabel('Coefficient of Variation')
plt.title('Figure 3: NDVI Spatial Heterogeneity (1985-2022, Excluding Lakes)', pad=15)
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "figure3_ndvi_spatial_heterogeneity.pdf"), dpi=600, bbox_inches='tight')
plt.close()

# 图表 4: NDVI 概率密度（添加 1985-2022 年的总体概率密度）
ndvi_flat_data = []
for year in selected_years:
    ndvi_year = ndvi_yearly.sel(year=year).compute()
    ndvi_values = ndvi_year.values.flatten()
    ndvi_values = ndvi_values[~np.isnan(ndvi_values)]
    ndvi_flat_data.append({'Year': year, 'NDVI': ndvi_values})

# 计算 1985-2022 年的总体概率密度
all_ndvi_values = []
for year in all_years:
    ndvi_year = ndvi_yearly.sel(year=year).compute()
    ndvi_values = ndvi_year.values.flatten()
    ndvi_values = ndvi_values[~np.isnan(ndvi_values)]
    all_ndvi_values.extend(ndvi_values)

plt.figure(figsize=(12, 6))
colors = sns.color_palette("tab10", len(selected_years))
for i, data in enumerate(ndvi_flat_data):
    year = data['Year']
    ndvi_values = data['NDVI']
    sns.kdeplot(ndvi_values, label=str(year), color=colors[i], linewidth=2.5)

# 添加总体概率密度
sns.kdeplot(all_ndvi_values, label='1985-2022', color='black', linestyle='--', linewidth=3)

plt.xlabel('NDVI')
plt.ylabel('Density')
plt.title('Figure 4: NDVI Probability Density (Actual Data, 1985-2022, Excluding Lakes)', pad=15)
plt.legend(title='Year', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "figure4_ndvi_probability_density.pdf"), dpi=600, bbox_inches='tight')
plt.close()

print("All plots and statistics have been generated successfully!")