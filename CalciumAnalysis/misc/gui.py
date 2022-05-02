#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import PySimpleGUI as sg

def gui_set_directory():
    sg.theme('DarkAmber')  
    
    left_col = [[sg.Text('Folder'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()]]
    layout = [[sg.Column(left_col, element_justification='c')]]    
    window = sg.Window('Multiple Format Image Viewer', layout,resizable=True)
         
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == '-FOLDER-':
            folder = values['-FOLDER-']
            print(folder)
            window.Close()
            
    return folder
        
        

