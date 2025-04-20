# 导入所需库
import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask.array as da
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl

# 设置全局绘图样式
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 15

# 设置文件路径
output_dir = "D:/BaiduNetdiskDownload/figures"
nc_mean_path = os.path.join(output_dir, "ndvi_mean_1985_2022.nc")
nc_yearly_path = os.path.join(output_dir, "ndvi_yearly_1985_2022.nc")
basin_shp_path = "E:/iCloudDrive/Desktop/博士期间/毕业论文/data/30子流域.shp"
LAKES_SHP = r'E:\iCloudDrive\Desktop\博士期间\论文\蓝藻水华\2023.12.01\data\sum_lake.shp'

# 创建输出目录（如果不存在）
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

# 加载 NetCDF 文件并启用 dask
try:
    ds_mean = xr.open_dataset(nc_mean_path, chunks={'lat': 100, 'lon': 100})
    ds_yearly = xr.open_dataset(nc_yearly_path, chunks={'year': 10, 'lat': 100, 'lon': 100})
except Exception as e:
    raise ValueError(f"Error loading NetCDF files: {e}")

# 打印 NetCDF 文件的变量和维度信息
print("ds_mean variables and dimensions:")
print(ds_mean)
print("\nds_yearly variables and dimensions:")
print(ds_yearly)

# 检查变量是否存在
if 'NDVI' not in ds_mean or 'NDVI' not in ds_yearly:
    raise KeyError("Variable 'NDVI' not found in one or both NetCDF files. Available variables in ds_mean: "
                   f"{list(ds_mean.variables)}, in ds_yearly: {list(ds_yearly.variables)}")

# 设置空间维度和 CRS
try:
    # 重命名维度
    ndvi_mean = ds_mean['NDVI'].rename({'lat': 'y', 'lon': 'x'})
    ndvi_yearly = ds_yearly['NDVI'].rename({'lat': 'y', 'lon': 'x'})
    
    # 设置空间维度和 CRS
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

# 检查数据有效性
print(f"NDVI Mean - NaN count: {np.isnan(ndvi_mean).sum().compute()}")
print(f"NDVI Yearly - NaN count: {np.isnan(ndvi_yearly).sum().compute()}")

# --- 1. 空间分布图：动态选择年份 ---
ndvi_bins = [0, 0.3, 0.6, 1.0]
ndvi_colors = ['#e0f3db', '#a8ddb5', '#2a7d3e']
ndvi_cmap = ListedColormap(ndvi_colors)
ndvi_norm = BoundaryNorm(ndvi_bins, len(ndvi_colors))

# 选择年份
available_years = ds_yearly['year'].values
target_years = [1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020]
selected_years = [year for year in target_years if year in available_years]
if 2022 not in selected_years:
    selected_years.append(2022)

print(f"Selected years: {selected_years}")

# 单独绘制每个年份的图
for year in selected_years:
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor('white')
    ndvi_year = ndvi_yearly.sel(year=year).compute()
    ndvi_year.plot(ax=ax, cmap=ndvi_cmap, norm=ndvi_norm, add_colorbar=False)
    basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
    ax.set_title(f'NDVI Spatial Distribution ({year})', pad=15)
    
    # 添加颜色条
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=ndvi_norm, cmap=ndvi_cmap), ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_ticks([0.15, 0.45, 0.8])
    cbar.set_ticklabels(['Low (<0.3)', 'Medium (0.3-0.6)', 'High (>0.6)'])
    cbar.set_label('NDVI', fontsize=12)
    
    plt.savefig(os.path.join(output_dir, f"ndvi_spatial_distribution_{year}.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

# 单独绘制均值图
plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_facecolor('white')
ndvi_mean.compute().plot(ax=ax, cmap=ndvi_cmap, norm=ndvi_norm, add_colorbar=False)
basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
ax.set_title('NDVI Mean (1985-2022)', pad=15)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=ndvi_norm, cmap=ndvi_cmap), ax=ax, orientation='horizontal', pad=0.05)
cbar.set_ticks([0.15, 0.45, 0.8])
cbar.set_ticklabels(['Low (<0.3)', 'Medium (0.3-0.6)', 'High (>0.6)'])
cbar.set_label('NDVI', fontsize=12)

plt.savefig(os.path.join(output_dir, "ndvi_mean_1985_2022.pdf"), dpi=300, bbox_inches='tight')
plt.close()

# --- 2. NDVI 趋势空间分布图（Sen 斜率） ---
def sen_slope_and_mk(data):
    if np.all(np.isnan(data)) or len(data) < 2:
        return np.nan, np.nan, np.nan
    result = mk.original_test(data)
    return result.slope, result.p, 1 if result.trend == 'increasing' else -1 if result.trend == 'decreasing' else 0

# 向量化计算 Sen 斜率和 MK 检验
def apply_sen_slope_and_mk(da):
    # 重新分块，确保 year 维度是一个单一的块
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

# 计算 Sen 斜率、p 值和趋势方向
sen_slope, p_value, trend_direction = apply_sen_slope_and_mk(ndvi_yearly)

# 转换为 DataArray
sen_slope_da = xr.DataArray(sen_slope, coords={'y': ndvi_yearly['y'], 'x': ndvi_yearly['x']}, dims=['y', 'x']).rio.write_crs("EPSG:4326")
p_values_da = xr.DataArray(p_value, coords={'y': ndvi_yearly['y'], 'x': ndvi_yearly['x']}, dims=['y', 'x']).rio.write_crs("EPSG:4326")
trend_directions_da = xr.DataArray(trend_direction, coords={'y': ndvi_yearly['y'], 'x': ndvi_yearly['x']}, dims=['y', 'x']).rio.write_crs("EPSG:4326")

# 掩膜
sen_slope_da = sen_slope_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)
p_values_da = p_values_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)
trend_directions_da = trend_directions_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)

trend_bins = [-0.01, -0.005, -0.0025, -0.0005, 0.0005, 0.0025, 0.005, 0.01]
trend_colors = ['#08519c', '#3182bd', '#6baed6', '#bdd7e7', '#eff3ff', '#fee0d2', '#fc9272', '#de2d26']
trend_cmap = ListedColormap(trend_colors)
trend_norm = BoundaryNorm(trend_bins, len(trend_colors))

plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_facecolor('white')
sen_slope_da.plot(ax=ax, cmap=trend_cmap, norm=trend_norm, add_colorbar=False)
basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
ax.set_title('NDVI Trend (Sen Slope, 1985-2022)', pad=15)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=trend_norm, cmap=trend_cmap), ax=ax, orientation='horizontal', pad=0.05)
cbar.set_ticks(trend_bins)
cbar.set_ticklabels(['<-0.01', '-0.01', '-0.005', '-0.0025', '0.0025', '0.005', '0.01', '>0.01'])
cbar.set_label('NDVI Trend (NDVI/year)', fontsize=12)

plt.savefig(os.path.join(output_dir, "ndvi_sen_trend_spatial_distribution.pdf"), dpi=300, bbox_inches='tight')
plt.close()

# --- 3. NDVI 显著性空间分布图 ---
significant = p_values_da < 0.05
trend_class = xr.where((significant & (trend_directions_da == 1)), 2,
                       xr.where((significant & (trend_directions_da == -1)), 1, 0))
trend_class_da = trend_class.rio.write_crs("EPSG:4326")
trend_class_da = trend_class_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)

trend_class_bins = [-0.5, 0.5, 1.5, 2.5]
trend_class_colors = ['#d3d3d3', '#ff7f00', '#31a354']
trend_class_cmap = ListedColormap(trend_class_colors)
trend_class_norm = BoundaryNorm(trend_class_bins, len(trend_class_colors))

plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_facecolor('white')
trend_class_da.plot(ax=ax, cmap=trend_class_cmap, norm=trend_class_norm, add_colorbar=False)
basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
ax.set_title('NDVI Trend Significance (MK Test, 1985-2022)', pad=15)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=trend_class_norm, cmap=trend_class_cmap), ax=ax, orientation='horizontal', pad=0.05)
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(['No Significant Trend (p ≥ 0.05)', 'Significant Decrease (p < 0.05)', 'Significant Increase (p < 0.05)'])
cbar.set_label('Trend Significance', fontsize=12)

plt.savefig(os.path.join(output_dir, "ndvi_trend_significance_mk.pdf"), dpi=300, bbox_inches='tight')
plt.close()

# --- 4. 逐年平均 NDVI 时间序列图（分阶段） ---
ndvi_yearly_mean = ndvi_yearly.mean(dim=['y', 'x'], skipna=True).compute()
ndvi_df = pd.DataFrame({'Year': ndvi_yearly_mean['year'].values, 'NDVI': ndvi_yearly_mean.values})

# 定义阶段
stages = [
    (1985, 1990, '1985-1990', 'orange'),
    (1991, 2000, '1991-2000', 'green'),
    (2001, 2010, '2001-2010', 'blue'),
    (2011, 2022, '2011-2022', 'purple')
]

plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='NDVI', data=ndvi_df, marker='o', color='black', label='Yearly Mean NDVI')

# 全阶段趋势
mk_result = mk.original_test(ndvi_df['NDVI'])
slope, p_value = mk_result.slope, mk_result.p
sig_mark = '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
years = ndvi_df['Year'].values
trend_line = slope * (years - years[0]) + ndvi_df['NDVI'].values[0]
plt.plot(years, trend_line, color='red', linestyle='--', 
         label=f'1985-2022: Slope={slope:.4f}{sig_mark}')

# 分阶段趋势
for start, end, label, color in stages:
    stage_df = ndvi_df[(ndvi_df['Year'] >= start) & (ndvi_df['Year'] <= end)]
    if len(stage_df) > 1:
        mk_result = mk.original_test(stage_df['NDVI'])
        slope, p_value = mk_result.slope, mk_result.p
        sig_mark = '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        stage_years = stage_df['Year'].values
        trend_line = slope * (stage_years - stage_years[0]) + stage_df['NDVI'].values[0]
        plt.plot(stage_years, trend_line, color=color, linestyle='--', 
                 label=f'{label}: Slope={slope:.4f}{sig_mark}')

plt.xlabel('Year')
plt.ylabel('NDVI')
plt.title('Yearly Mean NDVI (1985-2022, Excluding Lakes)', pad=15)
plt.legend(loc='best')
#plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, "ndvi_yearly_mean_timeseries_sen_mk_stages.pdf"), dpi=300, bbox_inches='tight')
plt.close()

print("All plots have been generated successfully!")