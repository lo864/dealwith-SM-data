import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib as mpl
from pathlib import Path

# 设置全局绘图样式
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 15

# 文件路径
output_dir = Path("D:/helpMM/作图/夜间灯光/light-results")
nightlight_tif_dir = Path("D:/helpMM/iterateData/mock-light")
basin_shp_path = Path("D:/helpMM/作图/shp数据/用水数据统计的data/30子流域.shp")
lakes_shp_path = Path("D:/helpMM/作图/lake_shp/sum_lake.shp")

# 创建输出目录
output_dir.mkdir(parents=True, exist_ok=True)

# 读取shapefile并转换为WGS84
basin_shp = gpd.read_file(basin_shp_path).to_crs("EPSG:4326")
lakes_shp = gpd.read_file(lakes_shp_path).to_crs("EPSG:4326")

# 获取流域边界框并添加缓冲区
buffer = 0.1
minx, miny, maxx, maxy = basin_shp.total_bounds - [buffer, buffer, -buffer, -buffer]

# 读取TIFF文件
tif_files = sorted(nightlight_tif_dir.glob("*.tif"))
if not tif_files:
    raise FileNotFoundError(f"No TIFF files found in {nightlight_tif_dir}")

# 从文件名提取年份（文件名格式如 mock_DMSP1992.tif）
years = []
for f in tif_files:
    try:
        # 分割文件名，例如 mock_DMSP1992.tif → ['mock', 'DMSP1992']
        parts = f.stem.split('_')
        if len(parts) != 2 or not parts[1].startswith('DMSP'):
            raise ValueError(f"Unexpected filename format: {f.name}")
        # 提取 DMSP1992 的最后4位作为年份 → 1992
        year = int(parts[1][-4:])
        years.append(year)
    except (IndexError, ValueError) as e:
        raise ValueError(f"Cannot extract year from filename: {f.name}. Error: {str(e)}")

# 读取并合并TIFF文件
datasets = [rioxarray.open_rasterio(f, chunks={'x': 256, 'y': 256}) for f in tif_files]
ds = xr.concat(datasets, dim='time')
ds['time'] = pd.to_datetime(years, format='%Y')

# 选择第一个波段并设置CRS
nightlight_data = ds.sel(band=1).rio.set_spatial_dims(x_dim='x', y_dim='y')
nightlight_data = nightlight_data.rio.reproject("EPSG:4326")
nightlight_data = nightlight_data.where(nightlight_data >= 0, np.nan)

# --- 1. 空间分布图 ---
nightlight_bins = [0, 50, 100, 255]
nightlight_colors = ['#000000', '#00FF00', '#FFFF00', '#FFFFFF']
nightlight_cmap = ListedColormap(nightlight_colors)
nightlight_norm = BoundaryNorm(nightlight_bins, len(nightlight_colors))

target_years = [1992, 1997, 2002, 2007, 2012, 2017, 2022]
selected_years = [y for y in target_years if y in years]
if years[-1] not in selected_years:
    selected_years.append(years[-1])

for year in selected_years:
    plt.figure(figsize=(10, 8))
    ax = plt.axes()
    ax.set_facecolor('white')
    nightlight_year = nightlight_data.sel(time=str(year)).compute()
    nightlight_year.plot(ax=ax, cmap=nightlight_cmap, norm=nightlight_norm, add_colorbar=False)
    basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
    lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(f'Nightlight Spatial Distribution ({year})', pad=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=nightlight_norm, cmap=nightlight_cmap), 
                       ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_ticks([25, 75, 150])
    cbar.set_ticklabels(['Low', 'Medium', 'High'])
    cbar.set_label('Nightlight', fontsize=12)

    plt.savefig(output_dir / f"nightlight_spatial_distribution_{year}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

# 均值图
nightlight_mean = nightlight_data.mean(dim='time').compute()
plt.figure(figsize=(10, 8))
ax = plt.axes()
ax.set_facecolor('white')
nightlight_mean.plot(ax=ax, cmap=nightlight_cmap, norm=nightlight_norm, add_colorbar=False)
basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_title(f'Nightlight Mean ({years[0]}-{years[-1]})', pad=15)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=nightlight_norm, cmap=nightlight_cmap), 
                   ax=ax, orientation='horizontal', pad=0.05)
cbar.set_ticks([25, 75, 150])
cbar.set_ticklabels(['Low', 'Medium', 'High'])
cbar.set_label('Nightlight', fontsize=12)

plt.savefig(output_dir / f"nightlight_mean_{years[0]}_{years[-1]}.pdf", dpi=300, bbox_inches='tight')
plt.close()

# --- 2. 夜间灯光趋势空间分布图（Sen斜率） ---
def sen_slope_and_mk(data):
    if np.all(np.isnan(data)) or len(data) < 2:
        return np.nan, np.nan, np.nan
    result = mk.original_test(data)
    return result.slope, result.p, 1 if result.trend == 'increasing' else -1 if result.trend == 'decreasing' else 0

def apply_sen_slope_and_mk(da):
    da = da.chunk({'time': -1})
    result = xr.apply_ufunc(
        sen_slope_and_mk, da, input_core_dims=[['time']], output_core_dims=[[], [], []],
        vectorize=True, dask='parallelized', output_dtypes=[float, float, float]
    )
    return result

sen_slope, p_value, trend_direction = apply_sen_slope_and_mk(nightlight_data)

# 只使用 y 和 x 坐标，排除 time 坐标
spatial_coords = {'y': nightlight_data.coords['y'], 'x': nightlight_data.coords['x']}
sen_slope_da = xr.DataArray(sen_slope, coords=spatial_coords, dims=['y', 'x']).rio.write_crs("EPSG:4326")
p_values_da = xr.DataArray(p_value, coords=spatial_coords, dims=['y', 'x']).rio.write_crs("EPSG:4326")
trend_directions_da = xr.DataArray(trend_direction, coords=spatial_coords, dims=['y', 'x']).rio.write_crs("EPSG:4326")

# 裁剪到流域边界
sen_slope_da = sen_slope_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True)
p_values_da = p_values_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True)
trend_directions_da = trend_directions_da.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True)

trend_bins = [-0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1]
trend_colors = ['#0000FF', '#3333FF', '#6666FF', '#9999FF', '#FF9999', '#FF6666', '#FF3333', '#FF0000']
trend_cmap = ListedColormap(trend_colors)
trend_norm = BoundaryNorm(trend_bins, len(trend_colors))

plt.figure(figsize=(10, 8))
ax = plt.axes()
ax.set_facecolor('white')
sen_slope_da.plot(ax=ax, cmap=trend_cmap, norm=trend_norm, add_colorbar=False)
basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_title(f'Nightlight Trend (Sen Slope, {years[0]}-{years[-1]})', pad=15)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=trend_norm, cmap=trend_cmap), 
                   ax=ax, orientation='horizontal', pad=0.05)
cbar.set_ticks(trend_bins)
cbar.set_ticklabels(['<-0.1', '-0.1', '-0.05', '-0.02', '0.02', '0.05', '0.1', '>0.1'])
cbar.set_label('Nightlight Trend (Nightlight/year)', fontsize=12)

plt.savefig(output_dir / f"nightlight_sen_trend_spatial_distribution.pdf", dpi=300, bbox_inches='tight')
plt.close()

# --- 3. 夜间灯光显著性空间分布图 ---
significant = p_values_da < 0.05
trend_class = xr.where((significant & (trend_directions_da == 1)), 2,
                       xr.where((significant & (trend_directions_da == -1)), 1, 0))
trend_class_da = trend_class.rio.write_crs("EPSG:4326").rio.clip(basin_shp.geometry, basin_shp.crs, drop=True)

trend_class_bins = [-0.5, 0.5, 1.5, 2.5]
trend_class_colors = ['#D3D3D3', '#FF7F00', '#31A354']
trend_class_cmap = ListedColormap(trend_class_colors)
trend_class_norm = BoundaryNorm(trend_class_bins, len(trend_class_colors))

plt.figure(figsize=(10, 8))
ax = plt.axes()
ax.set_facecolor('white')
trend_class_da.plot(ax=ax, cmap=trend_class_cmap, norm=trend_class_norm, add_colorbar=False)
basin_shp.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
lakes_shp.boundary.plot(ax=ax, edgecolor='blue', linewidth=1)
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_title(f'Nightlight Trend Significance (MK Test, {years[0]}-{years[-1]})', pad=15)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=trend_class_norm, cmap=trend_class_cmap), 
                   ax=ax, orientation='horizontal', pad=0.05)
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(['No Significant Trend (p ≥ 0.05)', 'Significant Decrease (p < 0.05)', 'Significant Increase (p < 0.05)'])
cbar.set_label('Trend Significance', fontsize=12)

plt.savefig(output_dir / f"nightlight_trend_significance_mk.pdf", dpi=300, bbox_inches='tight')
plt.close()

# --- 4. 逐年平均夜间灯光时间序列图 ---
nightlight_yearly_mean = nightlight_data.mean(dim=['y', 'x'], skipna=True).compute()
nightlight_df = pd.DataFrame({'Year': nightlight_yearly_mean['time'].dt.year.values, 
                             'Nightlight': nightlight_yearly_mean.values})

stages = [
    (1992, 1997, '1992-1997', 'orange'),
    (1998, 2007, '1998-2007', 'green'),
    (2008, 2017, '2008-2017', 'blue'),
    (2018, years[-1], f'2018-{years[-1]}', 'purple')
]

plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Nightlight', data=nightlight_df, marker='o', color='black', 
             label='Yearly Mean Nightlight')

mk_result = mk.original_test(nightlight_df['Nightlight'])
slope, p_value = mk_result.slope, mk_result.p
sig_mark = '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
years_array = nightlight_df['Year'].values
trend_line = slope * (years_array - years_array[0]) + nightlight_df['Nightlight'].values[0]
plt.plot(years_array, trend_line, color='red', linestyle='--', 
         label=f'Trend (slope={slope:.3f}{sig_mark})')

plt.legend()
plt.title(f'Yearly Mean Nightlight Trend ({years[0]}-{years[-1]})')
plt.xlabel('Year')
plt.ylabel('Nightlight Intensity')
plt.grid(True)
plt.savefig(output_dir / f"nightlight_yearly_mean_timeseries.pdf", dpi=300, bbox_inches='tight')
plt.close()