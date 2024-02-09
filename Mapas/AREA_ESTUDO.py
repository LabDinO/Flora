import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd


# ___Load the shapefile or GeoJSON containing state boundaries and names__
shapefile_path = 'C:/Users/flora/OneDrive/Documentos/MESTRADO_UFSC/rotinas/python/BR_UF_2022/BR_UF_2022.shp'
gdf = gpd.read_file(shapefile_path)

def main():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    #CORDENADAS
    ax.set_extent([-60, -30, -25, 8], crs=ccrs.PlateCarree())
    # Plot state names at centroid of each state
    for idx, row in gdf.iterrows():
        state_name = row['SIGLA_UF']  # Replace with the actual column name
        centroid = row['geometry'].centroid
        #plt.text(centroid.x, centroid.y, state_name, fontsize=12, fontweight='bold', ha='center', va='center')


        # Check if the centroid is within the extent
        if (-60 <= centroid.x <= -30) and (-25 <= centroid.y <= 0):
            plt.text(centroid.x, centroid.y, state_name, fontsize=9, fontweight='bold', ha='center', va='center')



    # Put a background image on for nice sea rendering.
    ax.stock_img()

    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')


    ax.add_feature(cfeature.STATES, linestyle=':')
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')

    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    #Add states borders
    #ax.add_feature(states_provinces , edgecolor='gray', linestyle=':') #
    #Add countrys borders
    ax.add_feature(cfeature.BORDERS, linestyle=':')

 # Add gridlines with labels on the left and bottom
    gl = ax.gridlines(draw_labels=False, dms=True, x_inline=False, y_inline=False)
    gl.right_labels = True 
    gl.bottom_labels = True 
    gl.xlabel_style = {'fontsize': 16}  # Adjust the fontsize as needed
    gl.ylabel_style = {'fontsize': 16}  # Adjust the fontsize as needed


#Vamos adcionar os pontos do arquivo csv com os avistamenos acústicos
    import pandas as pd
    import csv
    pontos_RN = pd.read_csv("C:/Users/flora/OneDrive/Documentos/MESTRADO_UFSC/rotinas/python/avistamentos_RN.csv")
    pontos_114 = pd.read_csv("C:/Users/flora/OneDrive/Documentos/MESTRADO_UFSC/rotinas/python/avistamentos_114.csv", delimiter=",")
    pontos_120 = pd.read_csv("C:/Users/flora/OneDrive/Documentos/MESTRADO_UFSC/rotinas/python/avistamentos_120.csv", delimiter=",")
# Plotar pontos no mapa
    ax.scatter(pontos_RN['Longitude'], pontos_RN['Latitude'], color='green', marker='o', s=20,linewidths=0.1, edgecolors='black',transform=ccrs.PlateCarree(), label='Projeto 1')
    ax.scatter(pontos_114['Longitude'], pontos_114['Latitude'], color='yelLow', marker='o', s=20,linewidths=0.1, edgecolors='black', transform=ccrs.PlateCarree(), label='Projeto 2')
    ax.scatter(pontos_120['Longitude'], pontos_120['Latitude'], color='red', marker='o', s=20,linewidths=0.1, edgecolors='black', transform=ccrs.PlateCarree(), label='Projeto 3')


# Adicione uma legenda
    # Adicione uma legenda e ajuste sua posição para o canto inferior
    ax.legend(loc='lower left', fontsize=13)


# Add title to the map
    ax.set_title('Pontos de Detecção Acústica', fontsize=20)


    # Save the figure with transparent background
    plt.savefig('C:/Users/flora/OneDrive/Documentos/MESTRADO_UFSC/rotinas/python/Area_estudo.png', dpi=plt.gcf().dpi)


    plt.show()


if __name__ == '__main__':
    main()