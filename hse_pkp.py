# import shapely
from urllib.parse import parse_qsl
import geopandas as gpd
import pandas as pd
from pyproj import Geod, crs, Transformer
import shapely
import re
import json
import sqlalchemy
from zipfile import ZipFile
import os
import rasterio
import rasterio.features
import numpy as np
from tqdm import tqdm
import math

# Сложности https://github.com/astral-sh/uv/issues/11466
from osgeo import gdal
from osgeo import gdal_array
from osgeo_utils import gdal_calc


def calculate_geod_buffers(
    i_gdf: gpd.GeoDataFrame,
    buffer_crs: str,
    buffer_dist_source: str,
    buffer_distance
):
    """_summary_

    Args:
        i_gdf (gpd.GeoDataFrame): входной набор данных
        buffer_crs (str): проекция для расчета буфера: 'utm' или 'laea'
        buffer_dist_source (str): источник значения размера буфера: 'field' (столбец в датафрейме) или 'value' (фикс значение)
        buffer_distance: в зависимости от значения buffer_dist_source - название столбца (str) или фикс значение (float)

    Raises:
        ValueError: Если Некорректно задан параметр buffer_crs

    Returns:
        list: Список с геометриями буферов
    """
    # Расчет буферов вокруг линейных водных объектов
    buffer_geom = []
    for i, row in i_gdf.iterrows():   # итерация по линейным объектам
        # формирование проекции UTM с центральным меридианом в центроиде текущего объекта
        lon = row['geometry'].centroid.x
        lat = row['geometry'].centroid.y
        if buffer_crs == 'utm':
            buf_crs = crs.CRS.from_proj4(
                f"+proj=tmerc +lat_0=0 +lon_0={lon} " \
                f"+k=0.9996 +x_0=500000 +y_0=0 +ellps=WGS84 " \
                f"+units=m +no_defs +type=crs"
            )
        elif buffer_crs == 'laea':
            buf_crs = crs.CRS.from_proj4(
                f"+proj=laea +lat_0={lat} +lon_0={lon} " \
                f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            )
        else:
            raise ValueError("Некорректно задан параметр buffer_crs")
        # модели пересчета туда и обратно
        transformer1 = Transformer.from_crs(4326, buf_crs, always_xy=True)
        transformer2 = Transformer.from_crs(buf_crs, 4326, always_xy=True)
        # вычисление буфера в текущей проекции UTM
        if buffer_dist_source == 'value':
            buffer = shapely.buffer(shapely.transform(row['geometry'], transformer1.transform, interleaved=False), float(buffer_distance))
        elif buffer_dist_source == 'field':
            buffer = shapely.buffer(shapely.transform(row['geometry'], transformer1.transform, interleaved=False), float(row[buffer_distance]))
        else:
            raise ValueError("Неверно задан параметр buffer_dist_source")
        # Пересчет буфера обратно в WGS-1984
        buffer = shapely.transform(buffer, transformer2.transform, interleaved=False)
        # Добавить результат в список
        buffer_geom.append(buffer)
    return buffer_geom


def prepare_water_limitations(
    source_water_line='data/water_line.shp',
    source_water_pol='data/water_pol.shp',
    result_gpkg='result/water_limitations.gpkg',
    buffer_distance = 5,
    buffer_crs = 'utm'
):
    """Функция рассчитывает ограничения для проекта ПКП для линейных и площадных водных объектов по методике Жанны. 
    Документация дорабатывается.

    Args:
        source_water_line (str, optional): Линейные водные объекты, исходные данные. Defaults to 'data/water_line.shp'.
        source_water_pol (str, optional): Площадные водные объекты, исходные данные. Defaults to 'data/water_pol.shp'.
        result_gpkg (str, optional): относительный путь к Geopackage для сохранения результатов.
        buffer_distance (int, optional): расстояние первого буфера от линейных объектов в м. Будет рассчитан в UTM с осевым меридианом в центроиде объекта или в LAEA с центром в центроиде объекта. Defaults to 5.
        buffer_crs (str, optional): проекция для расчета буферов. 'utm' или 'laea'. Defaults to 'utm'.
    """
    try:
        gdf_l = gpd.read_file(source_water_line)
        gdf_p = gpd.read_file(source_water_pol)
    except:
        raise
    
    # сливаем геометрию по полям fclass и name
    gdf_l = gdf_l.dissolve(by=['fclass', 'name'], dropna=False)
    
    # Объединяем куски полилиний
    gdf_l['geometry'] = gdf_l['geometry'].line_merge()
    
    # пересчитать в WGS-84 для вычисления геодезических расстояний
    gdf_l = gdf_l.to_crs(4326)
    
    # рассчитать геодезические длины
    geod = Geod(ellps='WGS84')
    gdf_l['leng'] = [geod.geometry_length(row['geometry']) for i, row in gdf_l.iterrows()]    
    # gdf.to_file(result_gpkg, layer='water_line_dissolved')

    # разбить мульти полилинии на отдельные
    gdf_l = gdf_l.explode()
    
    # рассчитать геодезические длины после разбивки на отдельные линии
    gdf_l['leng'] = [geod.geometry_length(row['geometry']) for i, row in gdf_l.iterrows()]    
    # gdf.to_file(result_gpkg, layer='water_line_dissolved_exploded')
    
    # отфильтровать ручьи и каналы короче 10 км
    gdf_l = gdf_l.query('not (fclass in("stream", "canal") and leng < 10000)')
    # gdf_l.to_file(result_gpkg, layer='water_line_dissolved_exploded_filtered')
    
    # Расчет буферов вокруг линейных водных объектов
    buffer_geom = calculate_geod_buffers(gdf_l, buffer_crs, 'value', buffer_distance)
    
    # gdf_l['buffer_geometry'] = buffer_geom
    gdf_l_buffered = gdf_l.set_geometry(buffer_geom)
    my_columns = gdf_l_buffered.columns.tolist()
    # my_columns.remove('buffer_geometry')
    gdf_l_buffered = gdf_l_buffered[my_columns]
    # расчет значений размеров буферов
    gdf_l_buffered['buf'] = [50 if x <= 10000 else 100 if 10000 < x <= 50000 else 200 for x in gdf_l_buffered['leng']]
    gdf_l['buf'] = [50 if x <= 10 else 100 if 10000 < x <= 50000 else 200 for x in gdf_l['leng']]
    # gdf_l.to_file(result_gpkg, layer='water_lines_buf')
    # вычислить геодезические буферы по полю buf  
    buffer_geom = calculate_geod_buffers(gdf_l_buffered, buffer_crs, 'field', 'buf')
    wp2 = gdf_l_buffered.set_geometry(buffer_geom)
    # wp2.to_file(result_gpkg, layer='wp2')
    
    # ---------------------------------------
    # обработка полигональных водных объектов
    
    # перепроецировать в WGS-1984
    gdf_p = gdf_p.to_crs(4326)
    # убрать все wetland
    gdf_p = gdf_p.query('not fclass in("wetland")')
    
    # Вычисление "длин" площадных объектов
    # Сделать пространственное присоединение обработанных линейных объектов по пересечению
    gdf_l = gdf_l.clip(gdf_p)
    gdf_l['leng'] = [geod.geometry_length(row['geometry']) for i, row in gdf_l.iterrows()]
    # gdf_l.to_file(result_gpkg, layer='water_lines_clipped')
    gdf_p = gdf_p.sjoin(gdf_l, how='left')
    # Отсортировать результат по длине присоединенного объекта по убыванию
    gdf_p = gdf_p.sort_values(by='leng', ascending=False)
    # Убрать дубликаты, оставив только первый - с максимальным значением длины
    gdf_p = gdf_p.drop_duplicates(subset=['osm_id_left'], keep='first')
    # Удалить лишние столбцы и убрать суффиксы от присоединения
    columns_to_drop = gdf_p.filter(regex='_right').columns.tolist()
    gdf_p = gdf_p.drop(columns=columns_to_drop)
    gdf_p = gdf_p.drop(columns=['width'])
    new_column_names = {col: re.sub(r'_left', '', col.lower()) for col in gdf_p.columns}
    gdf_p = gdf_p.rename(columns=new_column_names)
    
    # Вычислить геодезические площади
    gdf_p['area_km'] = [abs(geod.geometry_area_perimeter(row['geometry'])[0]) / 1000000 for i, row in gdf_p.iterrows()]
    
    # Там где не присоединилась длина линейного объекта и площадь больше 0.5, вставить buf=50
    gdf_p['buf'] = [50 if all([x >= 0.5, y]) else z for x, y, z in zip(gdf_p['area_km'], gdf_p['leng'].isna(), gdf_p['buf'])]
    # gdf_p.to_file(result_gpkg, layer='water_poly_leng')
    # убрать объекты с пустыми значениями buf
    gdf_p = gdf_p[gdf_p['buf'].notnull()]
    # вычислить геодезические буферы по полю buf  
    buffer_geom = calculate_geod_buffers(gdf_p, buffer_crs, 'field', 'buf')
    wp1 = gdf_p.set_geometry(buffer_geom)
    # wp1.to_file(result_gpkg, layer='wp1')
    
    # Объединение двух итоговых слоев
    wp1_wp2 = pd.concat([wp1, wp2])
    # wp1_wp2.to_file('result/water_line.gpkg', layer='wp1_wp2')
    water_prot_zone = wp1_wp2.union_all()
    water_prot_zone = gpd.GeoDataFrame(gpd.GeoSeries(water_prot_zone))
    water_prot_zone = water_prot_zone.rename(columns={0:'geometry'}).set_geometry('geometry')
    water_prot_zone = water_prot_zone.set_crs('epsg:4326')
    water_prot_zone = water_prot_zone.explode()
    
    water_prot_zone.to_file(result_gpkg, layer='water_prot_zone')


def prepare_slope_limitations(
    region='', 
    slope_threshold = 12,
    regions_table='admin.hse_russia_regions', 
    fabdem_tiles_table='elevation.fabdem_v1_2_tiles',
    fabdem_zip_path=r"\\172.21.204.20\geodata\_PROJECTS\pkp\vm0047_prod\dem_fabdem",
    rescale=True,
    rescale_size=10
    ):
    
    try:
        with open('.secret/.gdcdb', encoding='utf-8') as f:
            pg = json.load(f)
    except:
        raise
    try:
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{pg['user']}:{pg['password']}@{pg['host']}:{pg['port']}/{pg['database']}",
            connect_args={
                "sslmode": "verify-full",
                "target_session_attrs": "read-write"
                # "sslcert": "/path/to/client.crt",
                # "sslkey": "/path/to/client.key",
                # "sslrootcert": "/path/to/ca.crt",
            },
        )
        pass
    except:
        raise
    try:
        sql = f"select * from {regions_table} where lower(region) = '{region.lower()}';"
        region_gdf = gpd.read_postgis(sql, engine)
        sql = f"select * from {fabdem_tiles_table} fbdm where ST_Intersects((select geom from {regions_table} where lower(region) = '{region.lower()}' limit 1), fbdm.geom);"
        tiles_gdf = gpd.read_postgis(sql, engine)
    except:
        raise
    pass

    # temp_driver = gdal.GetDriverByName('MEM')

    current_dir = os.getcwd()
    os.environ["PROJ_LIB"] = os.path.join(current_dir, '.venv', 'Lib', 'site-packages', 'osgeo', 'data', 'proj')
    fabdemdir = os.path.join(current_dir, 'fabdem')
    if not os.path.isdir(fabdemdir):
        os.mkdir(fabdemdir)
    final_gdf = None

    region_centr_lat = region_gdf.iloc[0]['geom'].centroid.y
    region_centr_lon = region_gdf.iloc[0]['geom'].centroid.x

    for i, row in tqdm(tiles_gdf.iterrows(), desc='tiles loop', total=tiles_gdf.shape[0]):
        lon = row['geom'].centroid.x
        lat = row['geom'].centroid.y
        
        # extract current Fabdem tile
        zipfilename = row['zipfile_name']
        filename = row['file_name'].replace("N0", "N")
        zippath = os.path.join(fabdem_zip_path, zipfilename)
        try:
            with ZipFile(zippath, 'r') as zObject:
                zObject.extract(filename, path=fabdemdir)
            # print(f"Successfully extracted '{file_to_extract}' to '{destination_directory}'")
        except:
            raise
        input_file = os.path.join(fabdemdir, filename)
        input_dem = gdal.Open(input_file)
        
        # # reproject       # Перепроецирование в итоге убрали
        # if input_dem:
        #     # # UTM:
        #     # target_crs = f"+proj=tmerc +lat_0=0 +lon_0={lon} " \
        #     #     f"+k=0.9996 +x_0=500000 +y_0=0 +ellps=WGS84 " \
        #     #     f"+units=m +no_defs +type=crs"
        #     # LAEA:
        #     target_crs = f"+proj=laea +lat_0={region_centr_lat} +lon_0={region_centr_lon} " \
        #                  f"+x_0=4321000 +y_0=3210000 +ellps=WGS84 +units=m +no_defs"
        #     # # LAEA EPSG:3035
        #     # target_crs = f"+proj=laea +lat_0=52 +lon_0=10 " \
        #     #              f"+x_0=4321000 +y_0=3210000 +ellps=WGS84 +units=m +no_defs"
            
        #     output_reproj = os.path.join(fabdemdir, filename.replace('.tif', '_reproj.tif'))
        #     gdal.Warp(output_reproj, input_dem, dstSRS=target_crs)
            
        #     # temp_ds = temp_driver.Create('', 1, 1, 1, gdal.GDT_Byte)
        #     # gdal.Warp(temp_ds, input_dem, dstSRS=target_crs)
        #     input_dem = None
        #     input_dem = gdal.Open(output_reproj)
        #     pass
        
        # Rescaling with improved options
        output_rescale = None
        if rescale:
            rescale_options = gdal.WarpOptions(
                # xRes=rescale_size,
                # yRes=rescale_size,
                xRes=rescale_size / (111320 * math.cos(math.radians(lat))),
                yRes=rescale_size / (111320 * math.cos(math.radians(lat))),
                resampleAlg='bilinear',
                outputType=gdal.GDT_Float32,  # Specify output data type
                creationOptions=['COMPRESS=LZW', 'TILED=YES'],  # Compression and tiling
                multithread=True  # Enable multithreading for better performance
            )
            output_rescale = os.path.join(fabdemdir, filename.replace('.tif', '_reproj_rescale.tif'))
            # input_dem = None
            # input_dem = gdal.Open(output_reproj)
            if input_dem is None:
                # print(f"Error: Could not open {output_reproj}")
                pass
            else:
                gdal.Warp(output_rescale, input_dem, options=rescale_options)
                input_dem = None
                input_dem = gdal.Open(output_rescale)
                # gdal.Warp(output_rescale, temp_ds, options=rescale_options)        
        
        # Calculate slope
        # input_dem = None
        # input_dem = gdal.Open(output_rescale)        
        output_slope = os.path.join(fabdemdir, filename.replace('.tif', '_slope.tif'))
        if input_dem is None:
            print(f"Error: Could not open {os.path.join(fabdemdir, filename)}")
        else:
            # https://gdal.org/en/stable/api/python/utilities.html#osgeo.gdal.DEMProcessing
            # далее несколько вариантов расчета вертикального масштаба при вычислении slope, подсказанные GPT-5 (low reasoning)
            # scale = 1             # Это если в метрах
            # scale = 111320        # Это если в градусах и без учета широты
            # scale = 111320 * math.cos(math.radians(lat))      # Это если в градусах и по упрощенной формуле только с учетом масштаба по долготе в зависимости от широты
            # scale = 111132.954 * math.cos(math.radians(lat))  # Это если в градусах и по упрощенной формуле с учетом масштаба по долготе в зависимости от широты, с уточненным коэффициентом
            scale = math.sqrt((111132.954 - (559.822 * math.cos(math.radians(2 * lat))) + 1.175 * math.cos(math.radians(4 * lat))) * (111132.954 * math.cos(math.radians(lat))))  # Это если в градусах, по самой точной формуле как среднее геометрическое масштабов по долготе и широте в зависимости от широты
            gdal.DEMProcessing(
                output_slope, input_dem, "slope", 
                computeEdges=True, slopeFormat="degree", scale=scale   # Это если в градусах
                )
            # print(f"Slope calculated and saved to {output_slope}")
            pass
        
        # Reclass slope
        output_slope_reclass = os.path.join(fabdemdir, filename.replace('.tif', '_slope_reclass.tif'))
        calc_expression = f"(A>={slope_threshold})*1"
        ds = gdal_calc.Calc(calc_expression, A=output_slope, outfile=output_slope_reclass, overwrite=True)
        pass
        
        # Vectorize reclassed slope
        # https://www.google.com/search?q=python+rasterio+polygonize&sca_esv=7550878c098e0420&ei=AqK5aIbtPLqbwPAPxvCE6A8&ved=0ahUKEwiG9ujCqL-PAxW6DRAIHUY4Af0Q4dUDCBA&uact=5&oq=python+rasterio+polygonize&gs_lp=Egxnd3Mtd2l6LXNlcnAiGnB5dGhvbiByYXN0ZXJpbyBwb2x5Z29uaXplMgUQABjvBTIFEAAY7wUyBRAAGO8FSMUYUPQGWMIRcAF4AZABAJgBhwGgAbIEqgEDNy4xuAEDyAEA-AEBmAIIoALCA8ICChAAGLADGNYEGEfCAggQABgHGAgYHsICCBAAGIAEGKIEwgILEAAYgAQYhgMYigXCAgYQABgIGB6YAwCIBgGQBgiSBwE4oAfaFrIHATe4B7kDwgcFMC43LjHIBxM&sclient=gws-wiz-serp
        with rasterio.open(output_slope_reclass) as src:
            image = src.read(1)
            transform = src.transform
            crs = src.crs
            # mask = (image != 0 and image != src.nodata)
            mask = (image == 1) & (image != src.nodata)
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(
                    rasterio.features.shapes(image, mask=mask, transform=transform) # Replace mask=None with your mask if needed
                )
            )
            geoms = list(results)
            gdf = gpd.GeoDataFrame.from_features(geoms)
            gdf = gdf.set_crs(crs)
            gdf = gdf.to_crs('EPSG:4326')
            # add current vector data to final geodataframe
            if i == 0:
                final_gdf = gdf                
            else:
                final_gdf = pd.concat([final_gdf, gdf], ignore_index=True)
            # gdf.to_file('result/slope_limitations.gpkg', layer=filename.replace('.tif', '_slope'))
        ds = None
        input_dem = None
        # Delete intermediate rasters
        for fl in [
            output_slope, 
            output_slope_reclass, 
            # output_reproj, 
            input_file, 
            output_rescale
        ]:
            try:
                os.remove(fl)
            except Exception as err:
                print(err)
    if not final_gdf.empty:
        # Clip result by region and save to output
        final_gdf = final_gdf.clip(region_gdf)
        final_gdf.to_file('result/slope_limitations.gpkg', layer=region)        


if __name__ == '__main__':
    # prepare_water_limitations(
    #     source_water_line='data/Lipetsk_water_lines_3857/Lipetsk_water_lines_3857.shp',
    #     source_water_pol='data/Lipetsk_water_poly_3857/Lipetsk_water_poly_3857.shp',
    #     result_gpkg='result/water_limitations.gpkg',
    #     buffer_distance = 5,
    #     buffer_crs = 'utm'
    #     )
    
    prepare_slope_limitations(
        region='Липецкая область',
        slope_threshold=12,
        fabdem_zip_path=r"\\172.21.204.20\geodata\_PROJECTS\pkp\vm0047_prod\dem_fabdem",
        rescale=True,
        rescale_size=10
    )
    