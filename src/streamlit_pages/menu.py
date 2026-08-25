"""Menu page"""
import json

import streamlit as st
from streamlit_pages.common import MENU_FILE_CONFIG, config_missing_message


st.title("Snack Stack Menu", text_alignment="center")
st.divider()

uploaded_file = st.session_state[MENU_FILE_CONFIG]
if uploaded_file is not None:
  menu_dict = json.load(uploaded_file)
  if menu_dict:
    menu_by_cuisine = {}
    for menu_entry in menu_dict:
      cuisine = menu_entry["Cuisine"]
      if cuisine not in menu_by_cuisine:
        menu_by_cuisine[cuisine] = [menu_entry]
      else:
        menu_by_cuisine[cuisine].append(menu_entry)

    columns = st.columns(2)
    for index, (key, value) in enumerate(menu_by_cuisine.items()):
      with columns[index % 2]:
        container = st.container(border=True, gap="xxsmall")
        container.subheader(key)
        for menu_entry in value:
          container.write(f"##### {menu_entry['Dish']} ${menu_entry['Price']}")
          container.write(f"({menu_entry['Dietary']})")
          container.write(f"{menu_entry['Description']}")
else:
  st.warning(f"Menu file missing, configure application in sidebar\n. {config_missing_message()}")
