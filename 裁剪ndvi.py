import xarray as xr
import geopandas as gpd
import rioxarray
import numpy as np
import os
from pathlib import Path

# 设置路径
raw_data_dir = "D:/BaiduNetdiskDownload/data/ndvi"
output_dir = "D:/BaiduNetdiskDownload/figures"
basin_shp_path = "E:/iCloudDrive/Desktop/博士期间/毕业论文/data/30子流域.shp"
nc_yearly_path = os.path.join(output_dir, "ndvi_yearly_1986_2022.nc")  # 更新文件名

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取流域 shapefile 并转换为 WGS84 坐标系
basin_shp = gpd.read_file(basin_shp_path).to_crs("EPSG:4326")

# 获取流域边界框并添加缓冲区
bounds = basin_shp.total_bounds  # [minx, miny, maxx, maxy]
buffer = 0.1
minx, miny, maxx, maxy = bounds[0] - buffer, bounds[1] - buffer, bounds[2] + buffer, bounds[3] + buffer
print("Bounds after buffer:", minx, miny, maxx, maxy)

# 收集所有年份的 NDVI 数据
ndvi_yearly_list = []
for year in range(1985, 2023):  # 调整为 1986-2022
    raw_file = os.path.join(raw_data_dir, f"Daily_Gap-filled_NDVI_{year}.nc4")
    if not os.path.exists(raw_file):
        print(f"Warning: File for year {year} not found: {raw_file}")
        continue

    # 加载每日 NDVI 数据
    ds = xr.open_dataset(raw_file, chunks={'time': 100, 'lat': 100, 'lon': 100})
    
    # 调试：打印数据集结构
    print(f"\nDataset for year {year}:")
    print(ds)
    print("Variables:", list(ds.variables))
    print("Dimensions:", list(ds.dims))
    print("NDVI dimensions:", ds['NDVI'].dims)

    # 检查 NDVI 变量是否存在
    if 'NDVI' not in ds:
        print(f"Error: 'NDVI' variable not found in {raw_file}")
        continue

    ndvi = ds['NDVI']

    # 确保维度名正确
    expected_dims = {'lat', 'lon', 'time'}
    actual_dims = set(ndvi.dims)
    if not expected_dims.issubset(actual_dims):
        print(f"Error: Expected dimensions {expected_dims} not found in NDVI. Actual dimensions: {actual_dims}")
        # 尝试重命名维度（如果维度名是大写）
        rename_dict = {}
        for dim in ndvi.dims:
            if dim.lower() == 'lat':
                rename_dict[dim] = 'lat'
            elif dim.lower() == 'lon':
                rename_dict[dim] = 'lon'
            elif dim.lower() == 'time':
                rename_dict[dim] = 'time'
        if rename_dict:
            print(f"Renaming dimensions: {rename_dict}")
            ndvi = ndvi.rename(rename_dict)

    # 设置空间维度和 CRS
    try:
        ndvi = ndvi.rio.set_spatial_dims(x_dim='lon', y_dim='lat').rio.write_crs("EPSG:4326")
    except Exception as e:
        print(f"Error setting spatial dims for year {year}: {e}")
        # 手动设置坐标和 CRS
        if 'lon' in ds and 'lat' in ds:
            ndvi = ndvi.assign_coords(lon=ds['lon'], lat=ds['lat'])
            ndvi.rio.write_crs("EPSG:4326", inplace=True)
        else:
            print(f"Error: Cannot set spatial dims for year {year}. Skipping.")
            continue

    # 裁剪到流域边界框
    ndvi = ndvi.sel(lon=slice(minx, maxx), lat=slice(maxy, miny))

    # 使用流域 shapefile 进行掩膜
    ndvi = ndvi.rio.clip(basin_shp.geometry, basin_shp.crs, drop=True, all_touched=True).rio.write_crs("EPSG:4326")

    # 计算年均 NDVI
    ndvi_yearly = ndvi.mean(dim='time', skipna=True)
    ndvi_yearly = ndvi_yearly.expand_dims({'year': [year]})

    ndvi_yearly_list.append(ndvi_yearly)
    print(f"Processed year: {year}")

# 合并所有年份的数据
if not ndvi_yearly_list:
    raise ValueError("No NDVI data processed. Please check the raw data files.")
ndvi_yearly = xr.concat(ndvi_yearly_list, dim='year')

# 保存为 NetCDF 文件
ndvi_yearly.to_netcdf(nc_yearly_path)
print(f"Yearly NDVI data saved to: {nc_yearly_path}")

# 计算 1986-2022 年的均值并保存
ndvi_mean = ndvi_yearly.mean(dim='year', skipna=True)
nc_mean_path = os.path.join(output_dir, "ndvi_mean_1986_2022.nc")
ndvi_mean.to_netcdf(nc_mean_path)
print(f"Mean NDVI data saved to: {nc_mean_path}")

# 检查生成的年份范围
ds = xr.open_dataset(nc_yearly_path)
print("Years in ndvi_yearly:", ds['year'].values)