import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="ReceiptScan Dashboard", layout="wide")

st.title("ReceiptScan Dashboard")

if 'history' not in st.session_state:
    st.session_state.history = []

sidebar = st.sidebar
sidebar.title("Menu")
option = sidebar.radio("Selecciona opcion:", ["Explorador de Tickets", "Procesar Ticket", "Historial"])

if option == "Explorador de Tickets":
    st.header("Explorador de Tickets")
    
    data_dir = BASE_DIR / "data"
    image_files = list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
    
    if image_files:
        selected_file = st.selectbox("Selecciona ticket:", [f.name for f in image_files])
        
        if selected_file:
            img_path = data_dir / selected_file
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(img_rgb, caption=selected_file, use_column_width=True)
            
            with col2:
                st.write(f"**Archivo:** {selected_file}")
                st.write(f"**Dimensiones:** {img.shape if img is not None else 'N/A'}")
                st.write(f"**Tamaño:** {img_path.stat().st_size / 1024:.1f} KB")
    else:
        st.info("No se encontraron tickets en la carpeta data/")

elif option == "Procesar Ticket":
    st.header("Procesar Ticket con OCR")
    
    data_dir = BASE_DIR / "data"
    image_files = list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
    
    selected_file = st.selectbox("Selecciona ticket a procesar:", [f.name for f in image_files], key="process_select")
    
    if st.button("Procesar"):
        with st.spinner("Procesando..."):
            from src.text_extractionOCR import procesar_ticket, exportar_json
            
            img_path = data_dir / selected_file
            resultados, datos = procesar_ticket(str(img_path), debug=True)
            output_path = exportar_json(datos)
            
            st.session_state.history.append({
                "ticket": selected_file,
                "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "datos": datos,
                "path": str(output_path)
            })
            
            st.success("Procesamiento completado!")
            
            st.subheader("Resumen")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tienda", datos.get('tienda', 'N/A')[:30] if datos.get('tienda') else 'N/A')
            with col2:
                st.metric("Fecha", datos.get('fecha', 'N/A'))
            with col3:
                st.metric("Total", f"{datos.get('total', 'N/A')} EUR")
            
            if datos.get('productos'):
                st.subheader("Productos detectados")
                productos_df = pd.DataFrame(datos['productos'])
                productos_df['precio'] = productos_df['precio'].apply(lambda x: f"{x:.2f} EUR" if x else "N/A")
                st.dataframe(productos_df, use_container_width=True)

elif option == "Historial":
    st.header("Historial de Procesamientos")
    
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"{item['ticket']} - {item['fecha']}"):
                st.write(f"**Tienda:** {item['datos'].get('tienda', 'N/A')}")
                st.write(f"**Fecha:** {item['datos'].get('fecha', 'N/A')}")
                st.write(f"**Total:** {item['datos'].get('total', 'N/A')} EUR")
                st.write(f"**Productos:** {len(item['datos'].get('productos', []))}")
                st.json(item['datos'], expanded=False)
    else:
        st.info("No hay historial. Procesa un ticket primero.")