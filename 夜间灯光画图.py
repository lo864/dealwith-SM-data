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
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
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
output_dir = "D:/helpMM/作图/夜间灯光/light-results"  # 修改输出目录路径
# 假设这里没有了NetCDF文件，直接从tif文件读取，先设置tif文件所在目录
nightlight_tif_dir = "D:/helpMM/作图/夜间灯光"
basin_shp_path = "D:/helpMM/作图/shp数据/用水数据统计的data/30子流域.shp"  # 假设流域边界文件路径不变
LAKES_SHP = r'D:\helpMM\作图\lake_shp\export_sumlake.shp'  # 假设湖泊边界文件路径不变

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 检查文件是否存在
for path in [basin_shp_path, LAKES_SHP]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist: {path}")

# 读取shapefile并转换为WGS84坐标系
basin_shp = gpd.read_file(basin_shp_path).to_crs("EPSG:4326")
lakes_shp = gpd.read_file(LAKES_SHP).to_crs("EPSG:4326")

# 获取流域边界框并添加缓冲区
bounds = basin_shp.total_bounds
buffer = 0.1
minx, miny, maxx, maxy = bounds[0] - buffer, bounds[1] - buffer, bounds[2] + buffer, bounds[3] + buffer
print(f"Bounds with buffer: [{minx}, {miny}, {maxx}, {maxy}]")

# 读取tif文件并合并成一个xarray.Dataset，这里假设tif文件时间顺序连续且文件名有规律
# 先获取所有tif文件路径
tif_files = [os.path.join(nightlight_tif_dir, f) for f in os.listdir(nightlight_tif_dir) if f.endswith('.tif')]
tif_files.sort()  # 按文件名排序，确保时间顺序正确
# 读取每个tif文件并合并
ds = xr.concat([rioxarray.open_rasterio(f) for f in tif_files], dim='time')
# 假设时间维度命名为'time'，这里需要根据实际情况调整，如果文件名有年份信息，可从文件名提取年份作为时间坐标
# 这里简单假设时间坐标是从1992开始逐年递增
ds['time'] = pd.date_range(start='1992-01-01', periods=len(tif_files), freq='Y')
# 选择数据的一个波段（假设只有一个有效波段），并设置空间维度和CRS
nightlight_data = ds.sel(band=1).rename({'y': 'lat', 'x': 'lon'}).rio.set_spatial_dims(x_dim='lon', y_dim='lat').rio.write_crs("EPSG:4326")

# 裁剪夜间灯光数据到流域范围
nightlight_data = nightlight_data.sel({'lon': slice(minx, maxx), 'lat': slice(maxy, miny)})
# 使用流域shapefile进行掩膜
nightlight_data = nightlight_data.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)
# 移除不合理值（这里假设夜间灯光数据不应为负，根据实际情况调整）
nightlight_data = nightlight_data.where(nightlight_data >= 0, np.nan)

# 检查数据有效性
print(f"Nightlight Data - NaN count: {np.isnan(nightlight_data).sum().compute()}")

# --- 1. 空间分布图：动态选择年份 ---
nightlight_bins = [0, 50, 100, 255]  # 根据夜间灯光数据范围调整分箱，这里是示例，需根据实际调整
nightlight_colors = ['#000000', '#00FF00', '#FFFF00', '#FFFFFF']  # 示例颜色，需根据实际调整
nightlight_cmap = ListedColormap(nightlight_colors)
nightlight_norm = BoundaryNorm(nightlight_bins, len(nightlight_colors))

# 选择年份
available_years = ds['time'].dt.year.values
target_years = [1992, 1997, 2002, 2007, 2012, 2017, 2022]  # 自定义选择年份，可调整
selected_years = [year for year in target_years if year in available_years]
if 2023 not in selected_years:  # 根据tif文件年份范围调整
    selected_years.append(2023)

print(f"Selected years: {selected_years}")

# 单独绘制每个年份的图
for year in selected_years:
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor('white')
    nightlight_year = nightlight_data.sel(time=pd.Timestamp(str(year))).compute()
    nightlight_year.plot(ax=ax, cmap=nightlight_cmap, norm=nightlight_norm, add_colorbar=False)
    basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
    ax.set_title(f'Nightlight Spatial Distribution ({year})', pad=15)

    # 添加颜色条
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=nightlight_norm, cmap=nightlight_cmap), ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_ticks([25, 75, 150])  # 根据分箱调整刻度，示例值
    cbar.set_ticklabels(['Low', 'Medium', 'High'])  # 示例标签，可调整
    cbar.set_label('Nightlight', fontsize=12)

    plt.savefig(os.path.join(output_dir, f"nightlight_spatial_distribution_{year}.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

# 单独绘制均值图
nightlight_mean = nightlight_data.mean(dim='time').compute()
plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_facecolor('white')
nightlight_mean.plot(ax=ax, cmap=nightlight_cmap, norm=nightlight_norm, add_colorbar=False)
basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
ax.set_title('Nightlight Mean (1992-2023)', pad=15)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=nightlight_norm, cmap=nightlight_cmap), ax=ax, orientation='horizontal', pad=0.05)
cbar.set_ticks([25, 75, 150])
cbar.set_ticklabels(['Low', 'Medium', 'High'])
cbar.set_label('Nightlight', fontsize=12)

plt.savefig(os.path.join(output_dir, "nightlight_mean_1992_2023.pdf"), dpi=300, bbox_inches='tight')
plt.close()

# --- 2. 夜间灯光趋势空间分布图（Sen斜率） ---
def sen_slope_and_mk(data):
    if np.all(np.isnan(data)) or len(data) < 2:
        return np.nan, np.nan, np.nan
    result = mk.original_test(data)
    return result.slope, result.p, 1 if result.trend == 'increasing' else -1 if result.trend == 'decreasing' else 0

# 向量化计算Sen斜率和MK检验
def apply_sen_slope_and_mk(da):
    # 重新分块，确保time维度是一个单一的块
    da = da.chunk({'time': -1})
    result = xr.apply_ufunc(
        sen_slope_and_mk,
        da,
        input_core_dims=[['time']],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float, float]
    )
    return result

# 计算Sen斜率、p值和趋势方向
sen_slope, p_value, trend_direction = apply_sen_slope_and_mk(nightlight_data)

# 转换为DataArray
sen_slope_da = xr.DataArray(sen_slope, coords={'lat': nightlight_data['lat'], 'lon': nightlight_data['lon']}, dims=['lat', 'lon']).rio.write_crs("EPSG:4326")
p_values_da = xr.DataArray(p_value, coords={'lat': nightlight_data['lat'], 'lon': nightlight_data['lon']}, dims=['lat', 'lon']).rio.write_crs("EPSG:4326")
trend_directions_da = xr.DataArray(trend_direction, coords={'lat': nightlight_data['lat'], 'lon': nightlight_data['lon']}, dims=['lat', 'lon']).rio.write_crs("EPSG:4326")

# 掩膜
sen_slope_da = sen_slope_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)
p_values_da = p_values_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)
trend_directions_da = trend_directions_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)

trend_bins = [-0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1]  # 根据夜间灯光趋势范围调整，示例值
trend_colors = ['#0000FF', '#3333FF', '#6666FF', '#9999FF', '#FF9999', '#FF6666', '#FF3333', '#FF0000']  # 示例颜色，可调整
trend_cmap = ListedColormap(trend_colors)
trend_norm = BoundaryNorm(trend_bins, len(trend_colors))

plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_facecolor('white')
sen_slope_da.plot(ax=ax, cmap=trend_cmap, norm=trend_norm, add_colorbar=False)
basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
ax.set_title('Nightlight Trend (Sen Slope, 1992-2023)', pad=15)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=trend_norm, cmap=trend_cmap), ax=ax, orientation='horizontal', pad=0.05)
cbar.set_ticks(trend_bins)
cbar.set_ticklabels(['<-0.1', '-0.1', '-0.05', '-0.02', '0.02', '0.05', '0.1', '>0.1'])
cbar.set_label('Nightlight Trend (Nightlight/year)', fontsize=12)

plt.savefig(os.path.join(output_dir, "nightlight_sen_trend_spatial_distribution.pdf"), dpi=300, bbox_inches='tight')
plt.close()

# --- 3. 夜间灯光显著性空间分布图 ---
significant = p_values_da < 0.05
trend_class = xr.where((significant & (trend_directions_da == 1)), 2,
                       xr.where((significant & (trend_directions_da == -1)), 1, 0))
trend_class_da = trend_class.rio.write_crs("EPSG:4326")
trend_class_da = trend_class_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True)

trend_class_bins = [-0.5, 0.5, 1.5, 2.5]
trend_class_colors = ['#D3D3D3', '#FF7F00', '#31A354']
trend_class_cmap = ListedColormap(trend_class_colors)
trend_class_norm = BoundaryNorm(trend_class_bins, len(trend_class_colors))

plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_facecolor('white')
trend_class_da.plot(ax=ax, cmap=trend_class_cmap, norm=trend_class_norm, add_colorbar=False)
basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())
ax.set_title('Nightlight Trend Significance (MK Test, 1992-2023)', pad=15)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=trend_class_norm, cmap=trend_class_cmap), ax=ax, orientation='horizontal', pad=0.05)
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(['No Significant Trend (p ≥ 0.05)', 'Significant Decrease (p < 0.05)', 'Significant Increase (p < 0.05)'])
cbar.set_label('Trend Significance', fontsize=12)

plt.savefig(os.path.join(output_dir, "nightlight_trend_significance_mk.pdf"), dpi=300, bbox_inches='tight')
plt.close()

# --- 4. 逐年平均夜间灯光时间序列图（分阶段） ---
nightlight_yearly_mean = nightlight_data.mean(dim=['lat', 'lon'], skipna=True).compute()
nightlight_df = pd.DataFrame({'Year': nightlight_yearly_mean['time'].dt.year.values, 'Nightlight': nightlight_yearly_mean.values})

# 定义阶段
stages = [
    (1992, 1997, '1992-1997', 'orange'),
    (1998, 2007, '1998-2007', 'green'),
    (2008, 2017, '2008-2017', 'blue'),
    (2018, 2023, '2018-2023', 'purple')
]

plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Nightlight', data=nightlight_df, marker='o', color='black', label='Yearly Mean Nightlight')

# 全阶段趋势
# 全阶段趋势
mk_result = mk.original_test(nightlight_df['Nightlight'])
slope, p_value = mk_result.slope, mk_result.p
sig_mark = '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
years = nightlight_df['Year'].values
trend_line = slope * (years - years[0]) + nightlight_df['Nightlight'].values[0]
plt.plot(years, trend_line, color='red', linestyle='--', 
         label=f'1992-2023: Slope={slope:.4f}{sig_mark}')

# 分阶段趋势
for start, end, label, color in stages:
    stage_df = nightlight_df[(nightlight_df['Year'] >= start) & (nightlight_df['Year'] <= end)]
    if len(stage_df) > 1:
        mk_result = mk.original_test(stage_df['Nightlight'])
        slope, p_value = mk_result.slope, mk_result.p
        sig_mark = '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
        stage_years = stage_df['Year'].values
        trend_line = slope * (stage_years - stage_years[0]) + stage_df['Nightlight'].values[0]
        plt.plot(stage_years, trend_line, color=color, linestyle='--', 
                 label=f'{label}: Slope={slope:.4f}{sig_mark}')

plt.xlabel('Year')
plt.ylabel('Nightlight')
plt.title('Yearly Mean Nightlight (1992-2023, Excluding Lakes)', pad=15)
plt.legend(loc='best')
# plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(os.path.join(output_dir, "nightlight_yearly_mean_timeseries_sen_mk_stages.pdf"), dpi=300, bbox_inches='tight')
plt.close()