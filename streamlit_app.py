import streamlit as st
import folium
from streamlit_folium import st_folium
import xml.etree.ElementTree as ET
import numpy as np
import skfuzzy as fuzz
import pandas as pd
import os
from math import radians, sin, cos, sqrt, atan2

class EvacuationSystem:
    def __init__(self, kml_file_path=None, excel_filepaths=None):
        self.kml_file_path = kml_file_path
        self.excel_filepaths = excel_filepaths if excel_filepaths else []
        self.evacuation_paths = {}
        self.barangays = {
            "Poblacion": [10.78485548380185, 122.3837072230239], #1
            "Bobon": [10.89540587344769, 122.2974269217951], #2
            "Gines": [10.85687388394953, 122.3416150076236], #3
            "Barangbang": [10.81689999087547, 122.3308342242847], #4
            "Carolina": [10.79669908350887, 122.3520502173874], #5
            "Lonoc": [10.78288182778105, 122.3416334769778], #6
            "Bacolod": [10.87712542341233, 122.3069941457462], #7
            "Binolbog": [10.78306359215453, 122.3340049830963], #8
            "Ingay": [10.843979902549595, 122.2920820324205] #9
        }
        self.evacuation_center = "Poblacion"
        self.distances = {
            "Bobon": [24.4, 25.6], 
            "Gines": [12.7, 15.9,],
            "Bacolod": [19.9, 20.0],
            "Lonoc": [7.71, 10.6],
            "Binolbog": [8.74, 9.8],
            "Barangbang": [12.2, 17.0],
            "Carolina": [7.06, 7.23],
            "Ingay": [24.0, 28.7]
        }
        self.blocked_paths = {
            "Poblacion to Bobon",
            "Poblacion to Gines",
            "Poblacion to Barangbang",
            "Poblacion to Carolina",
            "Poblacion to Lonoc",
            "Poblacion to Binolbog",
            "Poblacion to Bacolod",
            "Poblacion to Ingay"
        }
        self.graph_connections = self._build_graph_connections()
        self.travel_data = {}
        if self.kml_file_path:
            self.load_kml_paths()
        if self.excel_filepaths:
            self.load_all_travel_data()

    def _build_graph_connections(self):
        return {
            "Poblacion": ["Bobon", "Gines", "Bacolod", "Lonoc", "Binolbog", 
                         "Barangbang", "Carolina", "Ingay"],
            "Bobon": ["Poblacion", "Gines", "Bacolod"],  
            "Gines": ["Poblacion", "Bobon", "Bacolod"],
            "Bacolod": ["Poblacion", "Bobon", "Gines", "Ingay"],
            "Lonoc": ["Poblacion", "Binolbog"],
            "Binolbog": ["Poblacion", "Lonoc"],
            "Barangbang": ["Poblacion", "Carolina"],
            "Carolina": ["Poblacion", "Barangbang"],
            "Ingay": ["Poblacion", "Bacolod"]
        }

    def haversine_distance(self, coord1, coord2):
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371
        return r * c

    def heuristic(self, node, goal):
        if node not in self.barangays or goal not in self.barangays:
            return float('inf')
        return self.haversine_distance(self.barangays[node], self.barangays[goal])

    def get_edge_cost(self, from_node, to_node):
        if to_node in self.travel_data:
            segments = self.travel_data[to_node]
            total_cost = 0
            total_time = 0
            for seg in segments:
                total_cost += self.fuzzy_evaluation(
                    seg['slope'], 
                    seg['travel_time'], 
                    seg['curvature']
                )
                total_time += seg['travel_time']
            return total_cost, total_time
        distance = self.haversine_distance(
            self.barangays[from_node], 
            self.barangays[to_node]
        )
        estimated_time = (distance / 40) * 60
        estimated_cost = distance * 0.5
        return estimated_cost, estimated_time

    def load_kml_paths(self):
        self.evacuation_paths = {}
        if not self.kml_file_path:
            return
        try:
            tree = ET.parse(self.kml_file_path)
            root = tree.getroot()
            namespace = ''
            if '}' in root.tag:
                namespace = root.tag.split('}')[0][1:]
            ns = {'kml': namespace}
            for placemark in root.findall('.//kml:Placemark', ns):
                name_element = placemark.find('kml:name', ns)
                coord_element = placemark.find('.//kml:coordinates', ns)
                if name_element is not None and coord_element is not None:
                    path_name = name_element.text.strip()
                    coord_text = coord_element.text.strip()
                    coord_points = []
                    for line in coord_text.split():
                        if line.strip():
                            try:
                                parts = line.strip().split(',')
                                if len(parts) >= 2:
                                    lon, lat = float(parts[0]), float(parts[1])
                                    coord_points.append([lat, lon])
                            except (ValueError, IndexError):
                                continue
                    if len(coord_points) >= 2:
                        self.evacuation_paths[path_name] = coord_points
            st.success(f"✅ Successfully loaded {len(self.evacuation_paths)} paths from KML file")
        except Exception as e:
            st.error(f"❌ Error loading KML file: {e}")

    def load_all_travel_data(self):
        loaded_count = 0
        for filepath in self.excel_filepaths:
            try:
                df = pd.read_excel(filepath)
                fname = os.path.basename(filepath)
                fname_without_ext = os.path.splitext(fname)[0]
                if "_to_" in fname_without_ext:
                    barangay_name = fname_without_ext.split("_to_")[-1]
                    barangay_name = barangay_name.replace("2", "").replace("3", "")
                    barangay_name = barangay_name.capitalize().strip()
                else:
                    continue
                slope_col = next((col for col in df.columns if "Slope" in col), None)
                time_col = next((col for col in df.columns if "Travel_Time_min" in col), None)
                if slope_col and time_col:
                    segment_data = []
                    for _, row in df.iterrows():
                        try:
                            dummy_curvature = np.random.uniform(0, 1)
                            segment_data.append({
                                "slope": float(row[slope_col]),
                                "travel_time": float(row[time_col]),
                                "curvature": dummy_curvature
                            })
                        except Exception:
                            continue
                    if segment_data:
                        self.travel_data[barangay_name] = segment_data
                        loaded_count += 1
            except Exception as e:
                st.warning(f"⚠️ Error processing file '{os.path.basename(filepath)}': {e}")
        if loaded_count > 0:
            st.success(f"✅ Loaded travel data for {loaded_count} barangays")

    def fuzzy_evaluation(self, slope, travel_time, curvature):
        x_slope = np.arange(-10, 11, 0.1)
        slope_low = fuzz.trimf(x_slope, [-10, -5, 0])
        slope_med = fuzz.trimf(x_slope, [-2, 0, 2])
        slope_high = fuzz.trimf(x_slope, [0, 5, 10])
        slope_level_low = fuzz.interp_membership(x_slope, slope_low, slope)
        slope_level_med = fuzz.interp_membership(x_slope, slope_med, slope)
        slope_level_high = fuzz.interp_membership(x_slope, slope_high, slope)

        x_time = np.arange(0, 31, 1)
        time_fast = fuzz.trimf(x_time, [0, 0, 10])
        time_avg = fuzz.trimf(x_time, [5, 15, 25])
        time_slow = fuzz.trimf(x_time, [20, 30, 30])
        time_level_fast = fuzz.interp_membership(x_time, time_fast, travel_time)
        time_level_avg = fuzz.interp_membership(x_time, time_avg, travel_time)
        time_level_slow = fuzz.interp_membership(x_time, time_slow, travel_time)

        x_curv = np.arange(0, 1.1, 0.01)
        curv_low = fuzz.trimf(x_curv, [0, 0, 0.5])
        curv_med = fuzz.trimf(x_curv, [0.2, 0.5, 0.8])
        curv_high = fuzz.trimf(x_curv, [0.5, 1, 1])
        curv_level_low = fuzz.interp_membership(x_curv, curv_low, curvature)
        curv_level_med = fuzz.interp_membership(x_curv, curv_med, curvature)
        curv_level_high = fuzz.interp_membership(x_curv, curv_high, curvature)

        cost_low = np.fmin(np.fmin(np.fmin(slope_level_low, time_level_fast), curv_level_low), 0.1)
        cost_med = np.fmin(np.fmax(np.fmax(slope_level_med, time_level_avg), curv_level_med), 0.5)
        cost_high = np.fmin(np.fmax(np.fmax(slope_level_high, time_level_slow), curv_level_high), 0.9)
        cost = np.fmax(cost_low, np.fmax(cost_med, cost_high))
        return cost

    def get_related_paths(self, barangay):
        base = f"{self.evacuation_center} to {barangay}"
        related_paths = []
        for path_name in self.evacuation_paths.keys():
            if path_name == base or path_name.startswith(base + "2") or path_name.startswith(base + "3"):
                related_paths.append(path_name)
        return related_paths

    def best_unblocked_path(self, barangay):
        candidates = []
        for path_name in self.get_related_paths(barangay):
            if path_name in self.blocked_paths:
                continue
            if barangay in self.distances:
                index = 0
                if "2" in path_name: index = 1
                elif "3" in path_name: index = 2
                if index < len(self.distances[barangay]):
                    dist = self.distances[barangay][index]
                else:
                    dist = float('inf')
            else:
                dist = float('inf')
            candidates.append((path_name, dist))
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1])[0]

    def create_evacuation_map(self, selected_barangay=None):
        map_center = self.barangays[self.evacuation_center]
        m = folium.Map(
            location=map_center, 
            zoom_start=12,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri World Imagery Satellite'
        )
        for name, coords in self.barangays.items():
            color = "red" if name == self.evacuation_center else "blue"
            if name == selected_barangay:
                color = "orange"
            folium.Marker(
                location=coords,
                popup=f"{name}{' (Evacuation Center)' if name == self.evacuation_center else ''}",
                icon=folium.Icon(color=color)
            ).add_to(m)
        if selected_barangay and selected_barangay != self.evacuation_center:
            related_paths = self.get_related_paths(selected_barangay)
            best_path_name = self.best_unblocked_path(selected_barangay)
            for path_name in related_paths:
                if path_name == best_path_name:
                    continue
                coords = self.evacuation_paths.get(path_name, None)
                if not coords:
                    continue
                if path_name in self.blocked_paths:
                    color, weight, opacity = "red", 3, 0.9  # thinner
                else:
                    color, weight, opacity = "gray", 3, 0.5
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=weight,
                    opacity=opacity,
                    tooltip=f"Path: {path_name}"
                ).add_to(m)
            if best_path_name:
                coords = self.evacuation_paths.get(best_path_name, None)
                if coords:
                    folium.PolyLine(
                        locations=coords,
                        color="blue",
                        weight=6,
                        opacity=0.9,
                        tooltip=f"{best_path_name} (best path)"
                    ).add_to(m)
        return m

def load_embedded_files():
    kml_file = None
    excel_files = []
    data_folder = "data"
    if os.path.exists(data_folder):
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            if file.endswith('.kml'):
                kml_file = file_path
            elif file.endswith(('.xlsx', '.xls')):
                excel_files.append(file_path)
    return kml_file, excel_files

def main():
    st.set_page_config(
        page_title="SAFE Route",
        page_icon="🚨",
        layout="wide"
    )
    st.title("🚨 SAFE Route (Smart Alert and Fast Evacuation with Rerouting Technology)")
    if 'system' not in st.session_state:
        st.session_state.system = None
        st.session_state.auto_loaded = False
    if not st.session_state.auto_loaded:
        kml_file, excel_files = load_embedded_files()
        if kml_file and excel_files:
            with st.spinner("Loading embedded data files..."):
                st.session_state.system = EvacuationSystem(kml_file, excel_files)
                st.session_state.auto_loaded = True
                st.success("✅ System ready with pre-loaded data!")
        else:
            st.session_state.auto_loaded = True
            st.error("❌ No data files found in 'data/' folder. Please ensure KML and Excel files are present.")
    with st.sidebar:
        st.header("ℹ️ System Information")
        if st.session_state.system and st.session_state.system.travel_data:
            st.metric("Barangays with Data", len(st.session_state.system.travel_data))
        else:
            st.warning("⚠️ No data loaded")
        st.markdown("---")
        st.markdown("### 📋 How to Use")
        st.markdown("""
        1. Select a barangay from the dropdown
        2. View the optimal evacuation route
        3. Check distance and travel time
        4. See the highlighted path on the map
        """)
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Edge costs based on:**
        - 🏔️ Road slope
        - ⏱️ Travel time
        - 🛣️ Path curvature
        **Color Legend:**
        - 🔴 Red: Blocked path
        - 🔵 Blue: Best available path
        - ⚪ Gray: Other available paths
        """)

    if st.session_state.system is None:
        st.warning("⚠️ System initialization failed")
        st.markdown("""
        ### 🚨 Data Files Not Found
        Please ensure the following:
        1. A `data/` folder exists in the project root
        2. The folder contains:
           - One `.kml` file (evacuation routes)
           - Multiple `.xlsx` files (travel data for each barangay)
        **Expected file structure:**
        ```
        project/
        ├── streamlit_app.py
        └── data/
            ├── completeroad.kml
            ├── Poblacion_to_Bacolod.xlsx
            ├── Poblacion_to_Carolina.xlsx
            └── ... (other Excel files)
        ```
        """)
    else:
        system = st.session_state.system
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("### Select Barangay")
            barangay_options = ["-- Select a barangay --"] + [
                name for name in system.barangays.keys() 
                if name != system.evacuation_center
            ]
            selected_barangay = st.selectbox(
                "Choose barangay to evacuate:",
                barangay_options,
                label_visibility="collapsed"
            )
        if selected_barangay and selected_barangay != "-- Select a barangay --":
            evacuation_map = system.create_evacuation_map(selected_barangay)
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("### 📊 Evacuation Details")
                st.markdown("**Based Speed: 40 kph**")
                best_path_name = system.best_unblocked_path(selected_barangay)
                best_index = None
                if best_path_name:
                    if "2" in best_path_name:
                        best_index = 2
                    elif "3" in best_path_name:
                        best_index = 3
                    else:
                        best_index = 1
                if best_path_name:
                    idx = 0
                    if "2" in best_path_name:
                        idx = 1
                    elif "3" in best_path_name:
                        idx = 2
                    if selected_barangay in system.distances and idx < len(system.distances[selected_barangay]):
                        dist = system.distances[selected_barangay][idx]
                    else:
                        dist = float('nan')
                    total_time = None
                    travel_barangay_name = selected_barangay
                    if "2" in best_path_name:
                        travel_barangay_name = selected_barangay
                    elif "3" in best_path_name:
                        travel_barangay_name = selected_barangay
                    if travel_barangay_name in system.travel_data:
                        total_time = sum(seg['travel_time'] for seg in system.travel_data[travel_barangay_name])
                    else:
                        total_time = dist / 40 * 60  # Use 40 kph
                    st.success("✅ Best Route Found ")
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("📏 Distance", f"{dist:.2f} km")
                    with metric_col2:
                        st.metric("⏱️ Travel Time", f"{total_time:.2f} min")
                else:
                    st.warning("⚠️ No unblocked path available for this barangay.")
                st.markdown("---")
                st.markdown("**Route Information:**")
                st.write(f"**From:** {system.evacuation_center} (Evacuation Center)")
                st.write(f"📍 **To: Brgy.** {selected_barangay}")
                if selected_barangay in system.distances:
                    st.markdown("**All Available Paths:**")
                    for i, dist in enumerate(system.distances[selected_barangay], 1):
                        tags = []
                        if i == 1 and f"{system.evacuation_center} to {selected_barangay}" in system.blocked_paths:
                            tags.append("blocked")
                        if i == best_index:
                            tags.append("best path")
                        tag_str = (" (" + ", ".join(tags) + ")") if tags else ""
                        st.write(f"Path {i}: {dist} km{tag_str}")
            with col2:
                st.markdown("**Map Name:** Esri World Imagery Satellite")
                st_folium(evacuation_map, width=900, height=600, returned_objects=[])
        else:
            default_map = system.create_evacuation_map()
            st.markdown("**Map Name:** Esri World Imagery Satellite")
            st_folium(default_map, width=1200, height=500, returned_objects=[])

    # Custom developer footer
    st.markdown("""
        <hr style='border:1px solid #bbb'>
        <div style='text-align:center; font-weight:bold; color:gray;'>
            Developed by Mel Elijah C. Amar &copy; 2025
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
