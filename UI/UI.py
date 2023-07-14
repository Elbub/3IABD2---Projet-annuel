#!/usr/bin/env python
# -*- encoding: utf-8 -*-

### Interface graphique
from tkinter import *
from tkinter import ttk
from tkinter.messagebox import *
from tkinter.filedialog import *
from screeninfo import get_monitors
import matplotlib.pyplot as plt

### Opérations sur les fichiers
import numpy as np
import os
from os.path import normpath
from PIL import Image
from json import dump, load

### Interops
import lib
import ctypes

### Divers
from typing import *
import re

### Debug
import sys
import time





### Constants
MAX_LAYER_NUMBER = 50
MAX_NUMBER_OF_CLASSES = 10

### Fonction de lecture des données de config
def read_config_data(file_name):
    """FR : Renvoie la dernière entrée du fichier texte indiqué. Les fichiers utilisant
    cette fonction ne doivent être modifiés que par les fonctions de cette application.

    EN : Returns the last entry in the indicated text file. Files using this function musn't
    be modified except by functions of this application."""
    with open(normpath(file_name), "r") as f:
        lines = f.readlines()
        data = lines[-1][11:-1]  # Datas are formated as "yyyy-mm-dd <data>\n".
    return data
#V

# Chemins des fichiers
CONFIG_FOLDER = read_config_data("UI/dossier_config_et_consignes.txt") + "/"
SAVE_FOLDER = read_config_data("UI/save_folder.txt")
LAYERS_SAVE_FOLDER = os.path.join(SAVE_FOLDER, "layers")
MODELS_SAVE_FOLDER = os.path.join(SAVE_FOLDER, "models")
ERRORS_SAVE_FOLDER = os.path.join(SAVE_FOLDER, "errors")


### Si besoin d'aide
def RTFM_protocol():
   """Most important protocol in IT : Read The Fancy Manual. Very useful when there's not enough functional neurons in the user's neural network."""
   os.startfile(r"README.md")
#V

### Fonctions gérant le défilement
def _bound_to_mousewheel(widget, event):
    """FR : Lie le défilement de la fenêtre à la molette de la souris lorsque le curseur est sur
    cette fenêtre.

    EN : Binds the window's scrolling to the mousewheel when the cursor is over that window.
    """
    widget.bind_all("<MouseWheel>", lambda e: _on_mousewheel(widget, e))


def _unbound_to_mousewheel(widget, event):
    """FR : Délie le défilement de la fenêtre à la molette de la souris lorsque le curseur sort
    de cette fenêtre.

    EN : Binds the window's scrolling to the mousewheel when the cursor leaves that window.
    """
    widget.unbind_all("<MouseWheel>")


def _on_mousewheel(widget, event):
    """FR : Fait défiler la fenêtre avec la molette.

    EN : Scrolls the window with the mousewheel."""
    widget.yview_scroll(int(-1 * (event.delta / 80)), "units")


### Fonctions de vérification des entrées des utilisateurs
def _check_entry_float(new_value):
   """FR : Empêche l'utilisateur d'entrer des valeurs incorrectes.
   
   EN : Prevent the user from entering incorrect values."""
   if new_value == "" :
      return True
   if re.match("^[0-9]+\.?[0-9]*$", new_value) is None and re.match("^[0-9]*\.?[0-9]+$", new_value) is None :
      return False
   return True
#V

def _check_entry_unsigned_int(new_value):
   """FR : Empêche l'utilisateur d'entrer des valeurs incorrectes.
   
   EN : Prevent the user from entering incorrect values."""
   if new_value == "" :
      return True
   if re.match("^[0-9]*$", new_value) is None :
      return False
   new_value = int(new_value)
   return new_value > 0
#V

def _check_entry_string(new_value):
   """FR : Empêche l'utilisateur d'entrer des valeurs incorrectes.
   
   EN : Prevent the user from entering incorrect values."""
   if new_value == "" :
      return True
   if re.match("^[-A-Za-z0-9éèêëàùôÎïÉÈÊËÀÙÔÏ_ ]*$", new_value) is None :
      return False
   return True
#V


##########################
def fonction_principale():
    
    def load_model_from_file():
        """FR : Ouvre une fenêtre de sélection de fichier pour choisir une liste de consignes à charger.
        
        EN : Opens a file selection window to choose a list of setpoints to load."""
        filepath = askopenfilename(title = "Load model", initialdir = MODELS_SAVE_FOLDER, filetypes = [("fichier json", "*.json")])
        if filepath :
            with open(filepath, 'r') as layers_file :
                nonlocal model
                potential_model = load(layers_file)
                if (not isinstance(potential_model, dict)  
                    or len(potential_model.keys() & set(["model_type", "layers", "weights", "accuracies_and_losses", "inputs_width", "inputs_height", "inputs_color", "dataset_folders", "is_classification"])) != 9):
                    showwarning("Wrong file", "No model found inside this file.")
                else :
                    model = potential_model
    #V
    
    def save_model_as_file():
        """FR : Ouvre une fenêtre de sélection de fichier pour choisir où enregistrer la liste de consignes actuelle.
        
        EN : Opens a file selection window to choose where to save the current list of setpoints."""
        
        filepath = asksaveasfilename(initialdir = MODELS_SAVE_FOLDER, filetypes = [("fichier json", "*.json")], defaultextension = ".json")
        if filepath :
            with open(filepath, 'w') as model_file :
                new_model = {}
                if isinstance(model["weights"], np.ndarray) :
                    new_model["weights"] = model["weights"].tolist()
                else :
                    new_model["weights"] = model["weights"]
                new_model["accuracies_and_losses"] = model["accuracies_and_losses"]
                new_model["model_type"] = model["model_type"]
                new_model["is_classification"] = model["is_classification"]
                new_model["inputs_width"] = model["inputs_width"]
                new_model["inputs_height"] = model["inputs_height"]
                new_model["inputs_color"] = model["inputs_color"]
                if isinstance(model["layers"], np.ndarray) :
                    new_model["layers"] = model["layers"].tolist()
                else :
                    new_model["layers"] = model["layers"]
                new_model["dataset_folders"] = model["dataset_folders"]
                dump(new_model, model_file)
            return True
    #V
    
    def update_model_display_frame():
        for widget in model_display_frame.winfo_children() :
            widget.destroy()
        Label(model_display_frame, text = "Model type :").grid(row = 0, column = 0, padx = 5, pady = 5)
        Label(model_display_frame, text = f"{model['model_type']} {model['layers'] if model['model_type'] == 'MLP' else ''}").grid(row = 0, column = 1, padx = 5, pady = 5)
        Label(model_display_frame, text = "Problem type :").grid(row = 1, column = 0, padx = 5, pady = 5)
        Label(model_display_frame, text = "Classification" if model['is_classification'] else "Regression").grid(row = 1, column = 1, padx = 5, pady = 5)
        Label(model_display_frame, text = "Input images size :").grid(row = 2, column = 0, padx = 5, pady = 5)
        Label(model_display_frame, text = f"{model['inputs_width']} x {model['inputs_height']}, {'gray' if model['inputs_color'] == 1 else 'RGB'}").grid(row = 2, column = 1, padx = 5, pady = 5)
    #V
    
    def model_configuration():
        """FR : Fenêtre de configuration des valeurs initiales de l'essai.
        
        EN : Test's initial values configuration window."""

        def empty_hidden_layers():
            """FR : Supprime toutes les consignes.
            
            EN : Deletes all the setpoints."""
            start_training_button["state"] = "normal",
            nonlocal layers
            layers = [layers[0], layers[-1]]
        #V
        
        def edit_layers():
            """FR : Fenêtre de configuration des consignes à suivre pendant l'essai.
            
            EN : Setup window for the setpoints to follow during the test."""
            def add_layer(new_layer_index):
                """FR : Évite de rajouter une consigne si l'utilisateur annule.
                
                EN : Avoids adding a setpoint if the user cancels."""
                nonlocal layers
                consigne_a_modifier = add_or_update_layer()
                        
                if consigne_a_modifier is not None :
                    layers.append(consigne_a_modifier)
                    for i in range (len(layers)-1, new_layer_index, -1) :
                        layers[i], layers[i - 1] = layers[i - 1], layers[i]
                    return layers_display_update()
            #V
            
            def update_layer(updating_layer_index):
                """FR : Évite de modifier la consigne si l'utilisateur annule.
                
                EN : Avoids modifying the setpoint if the user cancels."""
                nonlocal layers
                consigne_a_modifier = add_or_update_layer(updating_layer_index)
                if consigne_a_modifier is not None :
                    layers[updating_layer_index] = consigne_a_modifier
                    return layers_display_update()
            #V
            
            def add_or_update_layer(layer_being_edited: int = 0):
                """FR : Fenêtre permettant de définir ou modifier les paramètres de la consigne du type 
                selectionné. 
                
                EN : Window allowing to define or modify the parameters of the setpoint of the chosen type."""
                def validate_add_or_update():
                    """FR : Valide cette consigne et enregistre ses valeurs dans le dictionnaire à renvoyer.
                    
                    EN : Validates this setpoint and saves its values in the dictonnary to return."""
                    nonlocal validation, layer_edit_window
                    if not neurons_number.get():
                        showwarning("Incorrect number", "You must put at least 1 neuron per layer.")
                        layers_edit_window.lift()
                        layer_edit_window.lift()
                    else :
                        validation = True
                        layer_edit_window.destroy()
                #V
                validation = False
                is_an_update = layer_being_edited > 0 # False si ajout, True si modification
                layer_edit_window = Toplevel(layers_edit_window)
                layer_edit_window.grab_set()
                if is_an_update :
                    layer_edit_window.title("Update layer")
                else :
                    layer_edit_window.title("Add layer")
                
                neurons_number = IntVar()
                
                if is_an_update :
                    neurons_number.set(layers[layer_being_edited])
                Label(layer_edit_window, text = "Neurons number :").grid(row = 1, column = 0, padx = 5, pady = 5)
                Entry(layer_edit_window, width = 5, textvariable = neurons_number, validate = "key", validatecommand = (layer_edit_window.register(_check_entry_unsigned_int), '%P')).grid(row = 1, column = 1, padx = 5, pady = 5)

                Button(layer_edit_window, text = "Annuler", command = layer_edit_window.destroy).grid(row = 100, column = 0, columnspan = 2, padx = 5, pady = 5)
                Button(layer_edit_window, text = "Valider", command = validate_add_or_update).grid(row = 100, column = 2, columnspan = 3, padx = 5, pady = 5)
                
                layer_edit_window.wait_window()
                if validation :
                    return neurons_number.get()
            #V
        
            def cancel_changes():
                """FR : Annule les changements et restore la lsite de consignes précédente.
                
                EN : Cancels the changes and restores the list to its former state."""
                nonlocal layers
                layers = previous_layers
                layers_edit_window.destroy()
            #V
            
            def delete_layer(layer_index):
                """FR : Supprime la consigne donnée.
                
                EN : Deletes the given setpoint."""
                nonlocal layers
                layers.pop(layer_index)
                return layers_display_update()
            #V
            
            def load_layers_from_file():
                """FR : Ouvre une fenêtre de sélection de fichier pour choisir une liste de consignes à charger.
                
                EN : Opens a file selection window to choose a list of setpoints to load."""
                filepath = askopenfilename(title = "Load layers", initialdir = LAYERS_SAVE_FOLDER, filetypes = [("fichier json", "*.json")])
                if filepath :
                    with open(filepath, 'r') as layers_file :
                        nonlocal layers
                        potential_layers = load(layers_file)
                        if not isinstance(potential_layers, dict) or "layers" not in potential_layers.keys():
                            showwarning("Incorrect file : no layers found.")
                        else :
                            layers = potential_layers["layers"]
                    layers_display_update()
            #V
            
            def save_layers_as_file():
                """FR : Ouvre une fenêtre de sélection de fichier pour choisir où enregistrer la liste de consignes actuelle.
                
                EN : Opens a file selection window to choose where to save the current list of setpoints."""
                filepath = asksaveasfilename(initialdir = LAYERS_SAVE_FOLDER, filetypes = [("fichier json", "*.json")], defaultextension = ".json")
                if filepath :
                    with open(filepath, 'w') as layers_file :
                        dump({"layers" : layers}, layers_file)
            #V
            
            def layers_display_update():
                """FR : Gère l'affichage des consignes et de leurs boutons associés.
                
                EN : Manages the display of the setpoints and their associated buttons."""
                for widget in internal_layers_frame.winfo_children() :
                    widget.destroy()

                Button(internal_layers_frame, text = "Load from file", command = load_layers_from_file).grid(row = 0, column = 0, padx = 5, pady = 12)
                Button(internal_layers_frame, text = "Save in a file", command = save_layers_as_file).grid(row = 0, column = 2, padx = 5, pady = 12)
                
                Label(internal_layers_frame, text = "Current layers :").grid(row = 1, column = 0, columnspan = 3, padx = 5, pady = 4)
                
                current_layer_frame = LabelFrame(internal_layers_frame, text = f"Input layer")
                current_layer_frame.grid(row = 2, column = 0, columnspan = 3, padx = 5, pady = 4, sticky = 'w' + 'e')
                Label(current_layer_frame, text = f"{model['layers'][0]} neuron(s)").grid(row = 1, column = 0, padx = 5, pady = 4, sticky = 'w')
                if model_type.get() == "MLP" :
                    Button(internal_layers_frame, text = "Add layer", command = lambda : add_layer(1)).grid(row = 3, column = 1, padx = 5, pady = 5, sticky = 'e')
                        
                layer_index = 1
                for layer in layers[1:-1] :
                    layer_index += 1
                    
                    current_layer_frame = LabelFrame(internal_layers_frame, text = f"Hidden layer {layer_index - 1}")
                    current_layer_frame.grid(row = (2 * layer_index), column = 0, columnspan = 3, padx = 5, pady = 4, sticky = 'w' + 'e')
                    Label(current_layer_frame, text = f"{layer} neuron(s)").grid(row = 1, column = 0, padx = 5, pady = 4, sticky = 'w')
                    if model_type.get() == "MLP" :
                        Button(current_layer_frame, text = "Delete layer", command = lambda i = layer_index - 1 : delete_layer(i)).grid(row = 1, column = 1, padx = 5, pady = 5)
                    Button(current_layer_frame, text = "Update layer", command = lambda i = layer_index - 1 : update_layer(i)).grid(row = 1, column = 2, padx = 5, pady = 5, sticky = 'e')
                    if model_type.get() == "MLP" :
                        Button(internal_layers_frame, text = "Add layer", command = lambda i = layer_index : add_layer(i)).grid(row = (2 * layer_index + 1), column = 1, padx = 5, pady = 5, sticky = 'e')
                    internal_layers_frame.columnconfigure(0, weight = 1)
                    internal_layers_frame.columnconfigure(1, weight = 1)
                    internal_layers_frame.columnconfigure(2, weight = 1)
                
                current_layer_frame = LabelFrame(internal_layers_frame, text = f"Output layer")
                current_layer_frame.grid(row = (2 * MAX_LAYER_NUMBER), column = 0, columnspan = 3, padx = 5, pady = 4, sticky = 'w' + 'e')
                Label(current_layer_frame, text = f"{model['layers'][-1]} neuron(s)").grid(row = 1, column = 0, padx = 5, pady = 4, sticky = 'w')
                
                Button(internal_layers_frame, text = "Cancel", command = cancel_changes).grid(row = (2 * MAX_LAYER_NUMBER + 2), column = 0, padx = 5, pady = 5)
                if model_type.get() == "MLP" :
                    Button(internal_layers_frame, text = "Empty hidden layers", command = lambda : [empty_hidden_layers(), layers_display_update()]).grid(row = (2 * MAX_LAYER_NUMBER + 2), column = 1, padx = 5, pady = 5)
                Button(internal_layers_frame, text = "Validate", command = lambda : [layers_edit_window.destroy(), display_model_specs()]).grid(row = (2 * MAX_LAYER_NUMBER + 2), column = 2, padx = 5, pady = 5)
            #V

            nonlocal layers
            previous_layers = layers.copy()

            layers_edit_window = Toplevel(entries_window)
            layers_edit_window.grab_set()
            layers_edit_window.title("Fenetre de choix des consignes de l'essai")
            layers_edit_window.protocol("WM_delete_window", cancel_changes)
            layers_edit_window.rowconfigure(0, weight = 1)
            layers_edit_window.columnconfigure(0, weight = 1)
            layers_edit_window.columnconfigure(1, weight = 0)
            
            canevas = Canvas(layers_edit_window, width = 900, height = 500)
            canevas.grid(row = 0, column = 0, sticky = (N, S, E, W))
            canevas.rowconfigure(0, weight = 1)
            canevas.columnconfigure(0, weight = 1)
            y_scrollbar = ttk.Scrollbar(layers_edit_window, orient = "vertical", command = canevas.yview)
            y_scrollbar.grid(column = 1, row = 0, sticky = (N, S, E))
            internal_layers_frame = ttk.Frame(canevas, width = 880)
            internal_layers_frame.pack(in_ = canevas, expand = True, fill = BOTH)
            
            internal_layers_frame.bind("<Configure>", lambda _: canevas.configure(scrollregion = canevas.bbox("all")))
            canevas.create_window((0, 0), window = internal_layers_frame, anchor = "nw")
            canevas.configure(yscrollcommand = y_scrollbar.set)

            internal_layers_frame.bind('<Enter>', lambda e : _bound_to_mousewheel(canevas, e))
            internal_layers_frame.bind('<Leave>', lambda e : _unbound_to_mousewheel(canevas, e))

            layers_display_update()
            layers_edit_window.mainloop()
        #V
        
        def edit_number_of_clusters() :
            start_training_button["state"] = "disabled"
            empty_hidden_layers()
            nonlocal layers
            layers.append(1)
            layers[1], layers[2] = layers[2], layers[1]
        
        
        def display_model_specs():
            # Is okay
            """ """
            nonlocal layers
            for widget in model_type_frame.winfo_children()[4:] :
                widget.destroy()
            
            if model_type.get() == "Linear" :
                #TODO : check how I did the dic-filling window
                Label(model_type_frame, text = f'{layers[0]} input(s), {layers[-1]} output(s)').grid(row = 2, column = 0, columnspan = 3, padx = 10, pady = 10)
            elif model_type.get() == "MLP" :
                Label(model_type_frame, text = f'{layers[0]} input(s), {len(layers) - 2} hidden layers, {layers[-1]} output(s)').grid(row = 2, column = 0, columnspan = 3, padx = 10, pady = 10)
                Button(model_type_frame, text = "Edit layers", command = edit_layers).grid(row = 2, column = 3, padx = 10, pady = 10)
            elif model_type.get() == "SVM" :
                pass
                # Label(model_type_frame, text = f'{layers[0]} input(s), {len(layers) - 2} hidden layers, {layers[-1]} output(s)').grid(row = 2, column = 0, columnspan = 3, padx = 10, pady = 10)
            elif model_type.get() == "RBF" :
                Label(model_type_frame, text = f'{layers[0]} input(s), {len(layers) - 2} hidden layers, {layers[-1]} output(s)').grid(row = 2, column = 0, columnspan = 3, padx = 10, pady = 10)
                Button(model_type_frame, text = "Edit layers", command = edit_number_of_clusters).grid(row = 2, column = 3, padx = 10, pady = 10)
            else :
                pass
        #PV   facultatif

        def validate_entries():
            """FR : 
            
            EN : 
            """
            nonlocal model
            if (model["model_type"] != model_type.get()
                or (model["layers"].tolist() if isinstance(model["layers"], np.ndarray) else model["layers"]) != layers
                or model["is_classification"] != is_classification.get()
                or model["inputs_width"] != inputs_width.get()
                or model["inputs_height"] != inputs_height.get()
                or model["inputs_color"] != inputs_color.get()):
                model["weights"] = []
                model["accuracies_and_losses"] = []
            model["model_type"] = model_type.get()
            model["is_classification"] = is_classification.get()
            model["inputs_width"] = inputs_width.get()
            model["inputs_height"] = inputs_height.get()
            model["inputs_color"] = inputs_color.get()
            model["dataset_folders"] = [[classes_list[i][0].get(), classes_list[i][3].get()] for i in range(min(int(number_of_classes.get()) if number_of_classes.get() else 0, MAX_NUMBER_OF_CLASSES))]
            

            model["layers"] = layers
            entries_window.destroy()
            update_model_display_frame()
        #N
        
        def save_edited_model_as_file():
            """FR : Ouvre une fenêtre de sélection de fichier pour choisir où enregistrer la liste de consignes actuelle.
            
            EN : Opens a file selection window to choose where to save the current list of setpoints."""
            breaking_change = (model["model_type"] != model_type.get()
                            or model["layers"] != layers
                            or model["is_classification"] != is_classification.get()
                            or model["inputs_width"] != inputs_width.get()
                            or model["inputs_height"] != inputs_height.get()
                            or model["inputs_color"] != inputs_color.get())
            if breaking_change :
                showinfo("Info", "The model has been modified. No loss-related data will be saved.")
            filepath = asksaveasfilename(initialdir = MODELS_SAVE_FOLDER, filetypes = [("fichier json", "*.json")], defaultextension = ".json")
            if filepath :
                with open(filepath, 'w') as model_file :
                    new_model = {}
                    if breaking_change :
                        new_model["weights"] = []
                        new_model["accuracies_and_losses"] = []
                    else :
                        if isinstance(model["weights"], np.ndarray) :
                            new_model["weights"] = model["weights"].tolist()
                        else :
                            new_model["weights"] = model["weights"]
                        if isinstance(model["accuracies_and_losses"], np.ndarray) :
                            new_model["accuracies_and_losses"] = model["accuracies_and_losses"].tolist()
                        else :
                            new_model["accuracies_and_losses"] = model["accuracies_and_losses"]
                    new_model["model_type"] = model_type.get()
                    new_model["is_classification"] = is_classification.get()
                    new_model["inputs_width"] = inputs_width.get()
                    new_model["inputs_height"] = inputs_height.get()
                    new_model["inputs_color"] = inputs_color.get()
                    new_model["layers"] = layers
                    new_model["dataset_folders"] = [[classes_list[i][0].get(), classes_list[i][3].get()] for i in range(min(int(number_of_classes.get()) if number_of_classes.get() else 0, MAX_NUMBER_OF_CLASSES))]
                    dump(new_model, model_file)
            entries_window.lift()
        #V
        
        def compute_number_of_inputs():
            nonlocal layers
            if inputs_width.get() and inputs_height.get():
                layers[0] = int(inputs_width.get()) * int(inputs_height.get()) * inputs_color.get()
            display_model_specs()
        #V

        def set_last_layer():
            nonlocal layers
            if number_of_classes.get() :
                layers[-1] = int(number_of_classes.get())
            display_model_specs()
        #V
        
        def _check_entry_inputs(new_value):
            if _check_entry_unsigned_int(new_value):
                entries_window.after(1, compute_number_of_inputs)
                return True
            return False
        #V
        
        def _check_entry_classes(new_value):
            """FR : 
            
            EN : """
            def classes_display_update():
                """FR : Gère l'affichage des consignes et de leurs boutons associés.
                
                EN : Manages the display of the setpoints and their associated buttons."""
                for current_class_widgets in classes_list :
                    current_class_widgets[1].grid_forget()
                    current_class_widgets[2].grid_forget()
                    current_class_widgets[4].grid_forget()
                    current_class_widgets[5].grid_forget()

                for class_number in range(min(int(number_of_classes.get()) if number_of_classes.get() else 0, MAX_NUMBER_OF_CLASSES)) :
                    current_class_widgets = classes_list[class_number]
                    current_class_widgets[1].grid(row = class_number + 1, column = 0, padx = 5, pady = 5, sticky = W)
                    current_class_widgets[2].grid(row = class_number + 1, column = 1, padx = 5, pady = 5)
                    current_class_widgets[4].grid(row = class_number + 1, column = 2, padx = 5, pady = 5, sticky = E)
                    current_class_widgets[5].grid(row = class_number + 1, column = 3, padx = 5, pady = 5)
                    
            #V
            if _check_entry_unsigned_int(new_value) :
                if new_value :
                    if int(new_value) > MAX_NUMBER_OF_CLASSES :
                        showinfo("Too much classes", f"We currently support only up to {MAX_NUMBER_OF_CLASSES} classes.")
                    entries_window.after(1, set_last_layer)
                    entries_window.after(1, classes_display_update)
                return True
            return False
        #V

        nonlocal model
        
        entries_window = Toplevel(main_window)
        entries_window.grab_set()
        entries_window.title("Configuration initiale")
        entries_window.protocol("WM_DELETE_WINDOW", entries_window.destroy)
        model_type = StringVar()
        model_type.set(str(model["model_type"]))
        is_classification = BooleanVar()
        is_classification.set(bool(model["is_classification"]))
        inputs_width = StringVar()
        inputs_width.set(model["inputs_width"])
        inputs_height = StringVar()
        inputs_height.set(model["inputs_height"])
        inputs_color = IntVar()
        inputs_color.set(model["inputs_color"])
        number_of_classes = StringVar()
        number_of_classes.set(model["layers"][-1])
        if isinstance(model["layers"], np.ndarray) :
            layers = model["layers"].to_list()
        else :
            layers = model["layers"]
        
        model_type_frame = LabelFrame(entries_window, text = "Model type :")
        model_type_frame.grid(row = 1, columnspan = 7, column = 0, padx = 10, pady = 10)
        Radiobutton(model_type_frame, text = "Linear", variable = model_type, value = "Linear", command = lambda : [empty_hidden_layers(), display_model_specs()]).grid(row = 0, column = 0, padx = 5, pady = 5)
        Radiobutton(model_type_frame, text = "Multi-layer perceptron", variable = model_type, value = "MLP", command = lambda : [empty_hidden_layers(), display_model_specs()]).grid(row = 0, column = 1, padx = 5, pady = 5)
        # Radiobutton(model_type_frame, text = "Support vector machine", variable = model_type, value = "SVM", command = display_model_specs).grid(row = 0, column = 2, padx = 5, pady = 5)
        Radiobutton(model_type_frame, text = "Radial basis function", variable = model_type, value = "RBF", command = lambda : [edit_number_of_clusters(), display_model_specs()]).grid(row = 0, column = 3, padx = 5, pady = 5)
        
        Label( entries_window, text = "Task type :").grid(row = 2, column = 0, padx = 10, pady = 10)
        Radiobutton(entries_window, text = "Classification", variable = is_classification, value = True).grid(row = 2, column = 1, columnspan = 3, padx = 5, pady = 5)
        Radiobutton(entries_window, text = "Regression", variable = is_classification, value = False).grid(row = 2, column = 5, columnspan = 2, padx = 5, pady = 5)
        
        Label(entries_window, text = "Input image size :").grid(row = 3, column = 0, padx =10, pady =10)
        Entry(entries_window, textvariable = inputs_width, width = 5, validate = "key", validatecommand = (entries_window.register(_check_entry_inputs), '%P')).grid(row = 3, column = 1, padx = 0, pady = 5)
        Label(entries_window, text = " x ").grid(row = 3, column = 2, padx = 5, pady = 0)
        Entry(entries_window, textvariable = inputs_height, width = 5, validate = "key", validatecommand = (entries_window.register(_check_entry_inputs), '%P')).grid(row = 3, column = 3, padx = 0, pady = 5)
        Radiobutton(entries_window, text = "Grey", variable = inputs_color, value = 1, command = compute_number_of_inputs).grid(row = 3, column = 4, padx = 5, pady = 5)
        Radiobutton(entries_window, text = "RGB", variable = inputs_color, value = 3, command = compute_number_of_inputs).grid(row = 3, column = 5, padx = 5, pady = 5)
        
        internal_classes_frame = LabelFrame(entries_window, text = "Outputs :")
        internal_classes_frame.grid(row = 4, column = 0, columnspan = 10, padx = 10, pady = 10, sticky = NSEW)
        Label(internal_classes_frame, text = "Number of classes :").grid(row = 0, column = 0, padx = 5, pady = 5)
        Entry(internal_classes_frame, textvariable = number_of_classes, width = 5, validate = "key", validatecommand = (entries_window.register(_check_entry_classes), '%P')).grid(row = 0, column = 1, padx = 5, pady = 5, sticky = W)

        classes_list = []
        for class_number in range(MAX_NUMBER_OF_CLASSES) :
            current_class = [StringVar()]
            current_class[0].set(model["dataset_folders"][class_number][0] if len(model["dataset_folders"]) > class_number else "")
            current_class.append(Label(internal_classes_frame, text = f"Training dataset folder for class {class_number + 1} :"))
            current_class.append(Entry(internal_classes_frame, textvariable = current_class[0], width = 30, validate = "key"))
            current_class.append(StringVar())
            current_class[3].set(model["dataset_folders"][class_number][1] if len(model["dataset_folders"]) > class_number else "")
            current_class.append(Label(internal_classes_frame, text = f"Tests dataset folder for class {class_number + 1} :"))
            current_class.append(Entry(internal_classes_frame, textvariable = current_class[3], width = 30, validate = "key"))
            classes_list.append(current_class)
        
        
        Button(entries_window, text = "Cancel", command = entries_window.destroy).grid(row = 5, column = 0, padx =5, pady =5)
        Button(entries_window, text = "Load model", command = lambda : [entries_window.destroy(), load_model_from_file(), update_model_display_frame()]).grid(row = 5, column = 1, columnspan = 2, padx =5, pady =5)
        Button(entries_window, text = "Save model", command = save_edited_model_as_file).grid(row = 5, column = 3, columnspan = 2, padx =5, pady =5)
        Button(entries_window, text = "Ok", command = validate_entries).grid(row = 5, column = 5 , padx =5, pady =5)

        menubar = Menu(entries_window)
        entries_window.config(menu = menubar)
        menu = Menu(menubar, tearoff = 0)
        menubar.add_cascade(label = "Autre", menu = menu)
        menu.add_command(label = "Afficher la documentation", command = RTFM_protocol)

        entries_window.mainloop()
        
        return ()
    #V
    
    def save_and_quit():
        """FR : Fenêtre de choix des données à enregistrer avant de quitter ou relancer.

        EN : Window to choose which datas to save before quitting or launching another test.
        """

        def confirm_save_before_quiting():
            if save_model_as_file() :
                exit()
        #V
        
        def confirm_save_before_staying():
            if save_model_as_file() :
                program_close_window.destroy()
        #V
        
        program_close_window = Toplevel(main_window)
        program_close_window.grab_set()
        program_close_window.lift()

        Label(program_close_window, text = "Do you want to save your model ?").grid(row = 0, column = 0, columnspan = 3, padx = 10, pady = 10)
        Button(program_close_window, text = "Cancel", command = program_close_window.destroy).grid(row = 1, column = 0, padx = 10, pady = 10)
        Button(program_close_window, text = "Save and stay", command = confirm_save_before_staying).grid(row = 1, column = 1, padx = 10, pady = 10)
        Button(program_close_window, text = "Save and quit", command = confirm_save_before_quiting).grid(row = 1, column = 2, padx = 10, pady = 10)
        Button(program_close_window, text = "Quitter", command = exit).grid(row = 1, column = 3, padx = 10, pady = 10)
    #V
    
    # The following is a bunch of function that will be called upon pressing the UI's buttons.
    # Those are just skeletons, as of now. For each, we must :
    # - create adequate variables
    # - call the rust function
    # - delete the pointers if not needed anymore
    # - show some stuff, but not necessarily return anything

    def train_linear_model():
        pass
    #N
    
    def predict_with_linear_model():
        pass
    #N

    def train_mlp():
        """FR : 
        
        EN : """
        nonlocal current_train_inputs, current_train_labels, current_number_of_training_inputs, current_tests_inputs, \
            current_tests_labels, current_number_of_tests_inputs, current_learning_rate, current_number_of_epochs, current_batch_size
        current_learning_rate = float(learning_rate.get()) if learning_rate.get() else 0
        current_number_of_epochs = int(number_of_epochs.get()) if number_of_epochs.get() else 0
        current_batch_size = int(batch_size.get()) if batch_size.get() else 0
        if isinstance(current_tests_inputs, list) :
            current_train_inputs, current_train_labels, current_number_of_training_inputs, current_tests_inputs, current_tests_labels, current_number_of_tests_inputs \
                = lib.read_dataset(model["dataset_folders"])
        if isinstance(model["weights"], list) and model["weights"] == [] :
            model["weights"] = lib.generate_multi_layer_perceptron_model(model["layers"])
        model["weights"] = lib.train_multi_layer_perceptron_model(model["is_classification"],
                                                                  model["layers"],
                                                                  current_train_inputs,
                                                                  current_tests_inputs,
                                                                  current_train_labels,
                                                                  current_tests_labels,
                                                                  model["weights"],
                                                                  current_learning_rate,
                                                                  current_number_of_epochs,
                                                                  current_number_of_training_inputs,
                                                                  current_number_of_tests_inputs,
                                                                  model["layers"][0],
                                                                  len(model["dataset_folders"]),
                                                                  current_batch_size)
        #TODO : implement loss data retrieving. Something like `model[""] = load loss json` should do it
        with open("accuracies_and_losses.json", 'r') as accuracies_and_losses_file :
            model["accuracies_and_losses"].append(load(accuracies_and_losses_file))
        os.remove("accuracies_and_losses.json")
    #PV
    
    def predict_with_mlp(on_new_data : bool = False):
        """FR : 
        
        EN : """
        if on_new_data :
            resized_image = None
            filepath = askopenfilename(title = "Load model", initialdir = MODELS_SAVE_FOLDER, filetypes = [("fichier json", "*.jpg")])
            if filepath :
                print(filepath)
                image_file = Image.open(filepath)
                converted_image = image_file.convert("RGB" if model["inputs_color"] == 3 else "L")
                resized_image = converted_image.resize((int(model["inputs_width"]), int(model["inputs_height"])))
                if resized_image :
                    resized_image = np.array(resized_image, dtype = ctypes.c_float).flatten()  / 255 * 2 - 1
                else :
                    return
            else :
                return
        
        nonlocal current_train_inputs, current_train_labels, current_number_of_training_inputs, current_tests_inputs, current_tests_labels, current_number_of_tests_inputs
        if isinstance(current_tests_inputs, list) :
            current_train_inputs, current_train_labels, current_number_of_training_inputs, current_tests_inputs, current_tests_labels, current_number_of_tests_inputs \
                = lib.read_dataset(model["dataset_folders"])
        if isinstance(model["weights"], list) and model["weights"] == [] :
            model["weights"] = lib.generate_multi_layer_perceptron_model(model["layers"])
        number_of_inputs_for_this_test = 1 if on_new_data else current_number_of_tests_inputs
        predicted_dataset = lib.predict_with_multi_layer_perceptron_model(model["is_classification"],
                                                                        model["layers"],
                                                                        resized_image if on_new_data else current_tests_inputs,
                                                                        model["weights"],
                                                                        number_of_inputs_for_this_test,
                                                                        model["layers"][0],
                                                                        len(model["dataset_folders"]))
        predicted_dataset = predicted_dataset.reshape(number_of_inputs_for_this_test, len(model["dataset_folders"]))
        print(predicted_dataset)
        
        # The following is just a temporary and filthy way to see the errors.
        if on_new_data :
            number_of_predicted_classes = 0
            index_of_predicted_class = None
            for j in range(len(model["dataset_folders"])) :
                if predicted_dataset[0][j] > 0 :
                    index_of_predicted_class = j
                    number_of_predicted_classes += 1
            if number_of_predicted_classes != 1 :
                showinfo("Test failed", f"Couldn't find out of which class is this image.")
            else :
                showinfo("Test succeeded", f"This image is from class #{index_of_predicted_class + 1}.")
        else:
            labels_to_predict = current_tests_labels.reshape(number_of_inputs_for_this_test, len(model["dataset_folders"]))
            total_error = 0
            for i in range(number_of_inputs_for_this_test) :
                for j in range(len(model["dataset_folders"])) :
                    if abs(predicted_dataset[i][j] - labels_to_predict[i][j]) >= 1 :
                        total_error += 1
                        break
            showinfo("Test complete", f"Total number of error : {total_error}\
                    \nTotal number of inputs : {number_of_inputs_for_this_test}\
                    \nAccuracy : {round((1 - total_error / number_of_inputs_for_this_test) * 100, 1)}%")
    #N
    
    def train_model():
        #match case would be better, if venv were adequate (Python version >= 3.10)
        if model["model_type"] == "Linear" :
            train_linear_model()
        elif model["model_type"] == "MLP" :
            train_mlp()
        elif model["model_type"] == "SVM" :
            pass
        elif model["model_type"] == "RBF" :
            pass
    #V
    
    def predict_with_model():
        #match case would be better, if venv were adequate (Python version >= 3.10)
        if model["model_type"] == "Linear" :
            predict_with_linear_model()
        elif model["model_type"] == "MLP" :
            predict_with_mlp()
        elif model["model_type"] == "SVM" :
            pass
        elif model["model_type"] == "RBF" :
            pass
    #V
    
    def predict_on_given_dataset():
        predict_with_model()
    #V
    
    def print_loss():
        for number_of_training, training in enumerate(model["accuracies_and_losses"]) :
            fig, axs = plt.subplots(1, 3)
            current_training_title = f"Training {number_of_training} : "
            match model["model_type"] :
                case "Linear" :
                    pass
                case "MLP" :
                    current_training_title += f" with layers = {model['layers']}, on {training['number_of_epochs']} epochs."
                    pass
                case "RBF" :
                    pass
                case "SVM" :
                    pass
                case _ :
                    pass
            fig.suptitle(current_training_title)
            axs[0].plot(range(0, training["number_of_epochs"], training["batch_size"]), training["numbers_of_errors_on_training_dataset"], label = "Training")
            axs[0].plot(range(0, training["number_of_epochs"], training["batch_size"]), training["numbers_of_errors_on_tests_dataset"], label = "Tests")
            axs[0].set_title("Number of errors")
            axs[0].set(xlabel = "Epoch", ylabel = "Number of errors")
            axs[0].legend()
            axs[1].plot(range(0, training["number_of_epochs"], training["batch_size"]), [element * 100 for element in training["training_accuracies"]], label = "Training")
            axs[1].plot(range(0, training["number_of_epochs"], training["batch_size"]), [element * 100 for element in training["tests_accuracies"]], label = "Tests")
            axs[1].set_title("Accuracy")
            axs[1].set(xlabel = "Epoch", ylabel = "Accuracy")
            axs[1].legend()
            axs[2].plot(range(0, training["number_of_epochs"], training["batch_size"]), training["training_losses"], label = "Training")
            axs[2].plot(range(0, training["number_of_epochs"], training["batch_size"]), training["tests_losses"], label = "Tests")
            axs[2].set_title("Loss")
            axs[2].set(xlabel = "Epoch", ylabel = "Loss")
            axs[2].legend()
            
            plt.show()
    #V

    model = {"model_type" : "Linear",
             "layers" : [1, 1],
             "weights" : [],
             "accuracies_and_losses" : [],
             "inputs_width" : 1,
             "inputs_height" : 1,
             "inputs_color" : 1,
             "dataset_folders" : [["", ""]],
             "is_classification" : True}

    # each element in accuracies_and_losses is a dict looking like :
    # {
    #     "number_of_training_inputs" : int,
    #     "number_of_tests_inputs" : int,
    #     "number_of_epochs" : int,
    #     "batch_size" : int,
    #     "number_of_errors_on_training_dataset" : [];
    #     "training_accuracies" : [],
    #     "training_losses" : [],
    #     "number_of_errors_on_tests_dataset" : [];
    #     "tests_accuracies" : [],
    #     "tests_losses" : [],
    # }
    
    current_train_inputs = []
    current_train_labels = []
    current_number_of_training_inputs = 0
    current_tests_inputs = []
    current_tests_labels = []
    current_number_of_tests_inputs = 0
    current_learning_rate = 0
    current_number_of_epochs = 0
    current_batch_size = 0
    



    main_window = Tk()
    main_window.title("C'est la classe ! Mais laquelle ?")
    main_window.protocol("WM_DELETE_WINDOW", save_and_quit)

    ##### Organisation de l'affichage #####
    screen_list = get_monitors()
    screen_index = 0
    while (screen_index < len(screen_list) and not screen_list[screen_index].is_primary):
        screen_index += 1  # Cherche l'écran principal.
    screen_width = screen_list[screen_index].width
    screen_height = screen_list[screen_index].height
    # screen_width = 1440
    canvas = Canvas(main_window, height = int(screen_height / 2), width = screen_width / 3)
    canvas.grid(column = 0, row = 0, columnspan = 1, sticky = (N, W, E, S))
    width_scrollbar = ttk.Scrollbar(main_window, orient = HORIZONTAL, command = canvas.xview)
    width_scrollbar.grid(column = 0, row = 1, sticky = (W, E))
    height_scrollbar = ttk.Scrollbar(main_window, orient = VERTICAL, command = canvas.yview)
    height_scrollbar.grid(column = 1, row = 0, sticky = (N, S))
    cadre_interne = Frame(canvas)
    canvas.configure(xscrollcommand = width_scrollbar.set)
    canvas.configure(yscrollcommand = height_scrollbar.set)
    cadre_interne.bind("<Configure>", lambda _: canvas.configure(scrollregion = canvas.bbox("all")))
    canvas.create_window((0, 0), window = cadre_interne, anchor = "nw")

    cadre_interne.bind("<Enter>", lambda e : _bound_to_mousewheel(canvas, e))
    cadre_interne.bind("<Leave>", lambda e : _unbound_to_mousewheel(canvas, e))

    # Quand on modifie la taille de la fenêtre, la scrollbar reste de la même taille et
    # le reste s'agrandit.
    main_window.rowconfigure(0, weight = 1)
    main_window.rowconfigure(1, weight = 0)
    main_window.columnconfigure(0, weight = 1)
    main_window.columnconfigure(1, weight = 0)
    canvas.rowconfigure(0, weight = 1)
    canvas.columnconfigure(0, weight = 1)


    model_display_frame = LabelFrame(cadre_interne, text = "Model :")
    model_display_frame.grid(row = 0, column = 0, columnspan = 3, padx = 5, pady = 5, sticky = NSEW)
    update_model_display_frame()
    
    model_train_frame = LabelFrame(cadre_interne, text = "Training :")
    model_train_frame.grid(row = 0, column = 3, padx = 5, pady = 5, sticky = NSEW)
    learning_rate = StringVar()
    learning_rate.set("0.01")
    Label(model_train_frame, text = "Learning rate :").grid(row = 0, column = 0, padx = 5, pady = 5)
    Entry(model_train_frame, width = 5, textvariable = learning_rate, validate = "key", validatecommand = (model_train_frame.register(_check_entry_float), '%P')).grid(row = 0, column = 1, padx = 5, pady = 5)
    number_of_epochs = StringVar()
    number_of_epochs.set("100")
    Label(model_train_frame, text = "Number of epochs :").grid(row = 1, column = 0, padx = 5, pady = 5)
    Entry(model_train_frame, width = 5, textvariable = number_of_epochs, validate = "key", validatecommand = (model_train_frame.register(_check_entry_unsigned_int), '%P')).grid(row = 1, column = 1, padx = 5, pady = 5)
    batch_size = StringVar()
    batch_size.set("5")
    Label(model_train_frame, text = "Batch size :").grid(row = 2, column = 0, padx = 5, pady = 5)
    Entry(model_train_frame, width = 5, textvariable = batch_size, validate = "key", validatecommand = (model_train_frame.register(_check_entry_unsigned_int), '%P')).grid(row = 2, column = 1, padx = 5, pady = 5)
    start_training_button = Button(model_train_frame, text = "Train model", command = train_model)
    start_training_button.grid(row = 3, column = 0, columnspan = 2, padx = 5, pady = 5)
    
    model_predict_frame = LabelFrame(cadre_interne, text = "Exploiting :")
    model_predict_frame.grid(row = 1, column = 3, padx = 5, pady = 5, sticky = NSEW)
    Button(model_predict_frame, text = "Predict on given dataset", command = predict_on_given_dataset).grid(row = 0, column = 0, padx = 5, pady = 5)
    Button(model_predict_frame, text = "Predict on new data", command = lambda : predict_with_mlp(True)).grid(row = 1, column = 0, padx = 5, pady = 5)
    
    save_model_button = Button(cadre_interne, text = "Save model", command = save_model_as_file)
    save_model_button_image = PhotoImage(file = CONFIG_FOLDER + "icone_enregistrer.png")
    save_model_button.config(image = save_model_button_image)
    save_model_button.grid(row = 1, column = 0, padx = 5, pady = 5)
    Button(cadre_interne, text = "Edit model", command = lambda : [model_configuration(), update_model_display_frame()]).grid(row = 1, column = 1, padx = 5, pady = 5)
    print_loss_button = Button(cadre_interne, text = "Print loss", command = print_loss)
    print_loss_button_image = PhotoImage(file = CONFIG_FOLDER + "loss.png")
    print_loss_button.config(image = print_loss_button_image)
    print_loss_button.grid(row = 2, column = 3, columnspan = 1, padx = 5, pady = 5)
    Button(cadre_interne, text = "Save and quit", command = save_and_quit).grid(row = 2, column = 0, columnspan = 2, padx = 5, pady = 5)

    main_window.mainloop()

    return fonction_principale()


if __name__ == "__main__":
    fonction_principale()
