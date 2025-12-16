import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs

data = [
    ("Axiom-3 Dragon", 51.5, -104.0),
    ("Crew-7 Dragon", 35.5, -81.0),
    ("Crew-1 Dragon", -34.0, 149.0),
    ("Kenya debris", -2.0, 37.0),
    ("Argentina debris", -27.0, -61.0),
    ("Pilbara debris", -21.0, 119.0),
    ("Skylab debris", -31.0, 122.0),
    ("Kosmos-954", 65.0, -105.0),
    ("Long March 5B", 4.5, 114.0),
    ("Falcon 9 COPV", -33.5, 115.5),
    ("H-II debris", 31.5, 130.5),
    ("Point Nemo", -48.8, -123.4)
]
df = pd.DataFrame(data, columns=["Event","Latitude","Longitude"])

plt.figure(figsize=(12,6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.stock_img()
ax.coastlines(linewidth=0.8)

ax.scatter(df["Longitude"], df["Latitude"], color='red', s=60, edgecolor='black', zorder=5, label='Debris Event Locations')


for i, row in df.iterrows():
    ax.text(row["Longitude"], row["Latitude"]+2, row["Event"], fontsize=8, color='black', ha='center', va='bottom', zorder=10)

plt.title("Global Distribution of Confirmed Space Debris Impact / Recovery Locations")
plt.legend()
plt.show()


# For plotting in a longitude-latitude coordinates without projecting on the real map

# import matplotlib.pyplot as plt
# import pandas as pd

# data = [
#     ("Axiom-3 Dragon", 51.5, -104.0),
#     ("Crew-7 Dragon", 35.5, -81.0),
#     ("Crew-1 Dragon", -34.0, 149.0),
#     ("Kenya debris", -2.0, 37.0),
#     ("Argentina debris", -27.0, -61.0),
#     ("Pilbara debris", -21.0, 119.0),
#     ("Skylab debris", -31.0, 122.0),
#     ("Kosmos-954", 65.0, -105.0),
#     ("Long March 5B", 4.5, 114.0),
#     ("Falcon 9 COPV", -33.5, 115.5),
#     ("H-II debris", 31.5, 130.5),
#     ("Point Nemo", -48.8, -123.4)
# ]
# df = pd.DataFrame(data, columns=["Event","Latitude","Longitude"])

# plt.figure(figsize=(12,6))
# plt.scatter(df["Longitude"], df["Latitude"], color='red', s=60, edgecolor='black')
# for i, row in df.iterrows():
#     plt.text(row["Longitude"], row["Latitude"]+2, row["Event"], fontsize=8, ha='center', va='bottom', color='black')
# plt.xlabel("Longitude (°)")
# plt.ylabel("Latitude (°)")
# plt.title("Space Debris Impact/Recovery (No Base Map)")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("debris_points_only.png", dpi=150)
# plt.show()