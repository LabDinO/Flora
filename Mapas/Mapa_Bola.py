import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry.polygon import LinearRing
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from shapely.geometry.polygon import LinearRing
def sample_data(shape=(20, 30)):
    """
    Return ``(x, y, u, v, crs)`` of some vector data
    computed mathematically. The returned crs will be a rotated
    pole CRS, meaning that the vectors will be unevenly spaced in
    regular PlateCarree space.

    """
    #crs = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)



def main():
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-30,-10))



    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray')


    ax.set_global()
    ax.gridlines()


  #local visualizado  
    lons = [-50, -50, -30, -30]
    lats = [10,-10 ,-10, 10]
    ring = LinearRing(list(zip(lons, lats)))
    ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='red', edgecolor='black',alpha=0.8)

 # Save the figure with transparent background
    plt.savefig('C:/Users/flora/OneDrive/Documentos/MESTRADO_UFSC/rotinas/python/Mapa_Bola.png', transparent=True)

    plt.show()


if __name__ == '__main__':
    main()