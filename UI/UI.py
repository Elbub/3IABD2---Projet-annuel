#!/usr/bin/env python
# -*- encoding: utf-8 -*-

### Opérations sur les fichiers
import os
from os.path import normpath
from PIL import Image
from json import dump, load

### Interface graphique
from tkinter import *
from tkinter import ttk
from tkinter.messagebox import *
from tkinter.filedialog import *
from screeninfo import get_monitors
import matplotlib.pyplot as plt

### Divers
from typing import *
import re

### Debug
import sys
import time


### Constants
MAX_LAYER_NUMBER = 50

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

def RTM_protocol():
   """FR : Ouvre le manuel de la bibliothèque.

   EN : Opens the library's manual."""
   os.startfile(r"README.md")
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
                    or potential_model.keys() != ["model_type", "layers", "error_and_accuracy", "image_width", "image_height", "image_color", "dataset_folders", "is_classification"]):
                    showwarning("Incorrect file : no model found.")
                else :
                    model = potential_model
    #N
        
    
    def model_configuration () :
        """FR : Fenêtre de configuration des valeurs initiales de l'essai.
        
        EN : Test's initial values configuration window."""

        def edit_layers():
            """FR : Fenêtre de configuration des consignes à suivre pendant l'essai.
            
            EN : Setup window for the setpoints to follow during the test."""
            def add_layer(new_layer_index):
                """FR : Évite de rajouter une consigne si l'utilisateur annule.
                
                EN : Avoids adding a setpoint if the user cancels."""
                nonlocal model
                consigne_a_modifier = add_or_update_layer()
                        
                if consigne_a_modifier is not None :
                    model["layers"].append(consigne_a_modifier)
                    for i in range (len(model["layers"])-1, new_layer_index, -1) :
                        model["layers"][i], model["layers"][i - 1] = model["layers"][i - 1], model["layers"][i]
                    return actualisation_des_boutons()
                
            def update_layer(updating_layer_index):
                """FR : Évite de modifier la consigne si l'utilisateur annule.
                
                EN : Avoids modifying the setpoint if the user cancels."""
                nonlocal model
                consigne_a_modifier = add_or_update_layer(updating_layer_index)
                if consigne_a_modifier is not None :
                    model["layers"][updating_layer_index] = consigne_a_modifier
                    return actualisation_des_boutons()
            #V
            def add_or_update_layer(consigne_a_changer: int = 0):
                """FR : Fenêtre permettant de définir ou modifier les paramètres de la consigne du type 
                selectionné. 
                
                EN : Window allowing to define or modify the parameters of the setpoint of the chosen type."""
                def ajout_ou_modification_validee():
                    """FR : Valide cette consigne et enregistre ses valeurs dans le dictionnaire à renvoyer.
                    
                    EN : Validates this setpoint and saves its values in the dictonnary to return."""
                    nonlocal validation, layer_edit_window
                    validation = True
                    layer_edit_window.destroy()
                #V
                validation = False
                is_an_update = consigne_a_changer > 0 # False si ajout, True si modification
                layer_edit_window = Toplevel(layer_edit_window)
                if is_an_update :
                    layer_edit_window.title("Update layer")
                else :
                    layer_edit_window.title("Add layer")
                
                neurons_number = IntVar()
                if is_an_update :
                    neurons_number.set(consigne_a_changer)
                Label(layer_edit_window, text = "Neurons number :").grid(row = 1, column = 0, padx = 5, pady = 5)
                Entry(layer_edit_window, width = 5, textvariable = neurons_number, validate="key", validatecommand=(layer_edit_window.register(_check_entry_unsigned_int), '%P')).grid(row = 1, column = 1, padx = 5, pady = 5)

                Button(layer_edit_window, text = "Annuler", command = layer_edit_window.destroy).grid(row = 100, column = 0, columnspan = 2, padx = 5, pady = 5)
                Button(layer_edit_window, text = "Valider", command = ajout_ou_modification_validee).grid(row = 100, column = 2, columnspan = 3, padx = 5, pady = 5)
                
                layer_edit_window.wait_window()
                if validation :
                    return neurons_number.get()
            #V
            def empty_hidden_layers():
                """FR : Supprime toutes les consignes.
                
                EN : Deletes all the setpoints."""
                nonlocal model
                model["layers"] = [model["layers"][0], model["layers"][-1]]
                return actualisation_des_boutons()
            #V
            def cancel_changes():
                """FR : Annule les changements et restore la lsite de consignes précédente.
                
                EN : Cancels the changes and restores the list to its former state."""
                nonlocal model
                model["layers"] = previous_layers
                layer_edit_window.destroy()
            #V
            def delete_layer(layer_index):
                """FR : Supprime la consigne donnée.
                
                EN : Deletes the given setpoint."""
                nonlocal model
                model["layers"].pop(layer_index)
                return actualisation_des_boutons()
            #V
            def load_layers_from_file():
                """FR : Ouvre une fenêtre de sélection de fichier pour choisir une liste de consignes à charger.
                
                EN : Opens a file selection window to choose a list of setpoints to load."""
                filepath = askopenfilename(title = "Load layers", initialdir = LAYERS_SAVE_FOLDER, filetypes = [("fichier json", "*.json")])
                if filepath :
                    with open(filepath, 'r') as layers_file :
                        nonlocal model
                        potential_layers = load(layers_file)
                        if not isinstance(potential_layers, dict) or "layers" not in potential_layers.keys():
                            showwarning("Incorrect file : no layers found.")
                        else :
                            model["layers"] = potential_layers["layers"]
                    actualisation_des_boutons()
            #V
            def save_layers_as_file():
                """FR : Ouvre une fenêtre de sélection de fichier pour choisir où enregistrer la liste de consignes actuelle.
                
                EN : Opens a file selection window to choose where to save the current list of setpoints."""
                filepath = asksaveasfilename(initialdir = LAYERS_SAVE_FOLDER, filetypes = [("fichier json", "*.json")], defaultextension = ".json")
                if filepath :
                    with open(filepath, 'w') as layers_file :
                        dump({"layers" : model["layers"]}, layers_file)
            #V
            def actualisation_des_boutons():
                """FR : Gère l'affichage des consignes et de leurs boutons associés.
                
                EN : Manages the display of the setpoints and their associated buttons."""
                for widget in internal_layers_frame.winfo_children() :
                    widget.destroy()

                Button(internal_layers_frame, text = "Load from file", command = load_layers_from_file).grid(row = 0, column = 0, padx = 5, pady = 12)
                Button(internal_layers_frame, text = "Save in a file", command = save_layers_as_file).grid(row = 0, column = 2, padx = 5, pady = 12)
                
                Label(internal_layers_frame, text = "Current layers :").grid(row = 1, column = 0, columnspan = 3, padx = 5, pady = 4)
                
                current_layer_frame = LabelFrame(internal_layers_frame)
                current_layer_frame.grid(row = 2, column = 0, columnspan = 3, padx = 5, pady = 4, sticky = 'w' + 'e')
                Label(current_layer_frame, text = f"{model['layers'][0]} neuron(s)").grid(row = 1, column = 0, padx = 5, pady = 4, sticky = 'w')
                Label(internal_layers_frame, text = f"Input layer").grid(row = 2, column = 0, padx = 5, pady = 0, sticky = 'nw')
                Button(internal_layers_frame, text = "Add layer", command = lambda : add_layer(1)).grid(row = 3, column = 1, padx = 5, pady = 5, sticky = 'e')
                        
                layer_index = 1
                for layer in model["layers"][1:-1] :
                    layer_index += 1
                    
                    current_layer_frame = LabelFrame(internal_layers_frame)
                    current_layer_frame.grid(row = (2 * layer_index), column = 0, columnspan = 3, padx = 5, pady = 4, sticky = 'w' + 'e')
                    Label(current_layer_frame, text = f"{layer} neuron(s)").grid(row = 1, column = 0, padx = 5, pady = 4, sticky = 'w')
                    label_du_numero_de_bloc = Label(internal_layers_frame, text = f"Hidden layer {layer_index}")
                    label_du_numero_de_bloc.grid(row = (2 * layer_index), column = 0, padx = 5, pady = 0, sticky = 'nw')
                    # layer_edit_window.wm_attributes('-transparentcolor', label_du_numero_de_bloc['bg'])
                    # Label(current_layer_frame, image = PhotoImage(file = CONFIG_FOLDER + "rampe simple.png")).grid(row = 1, column = 0, padx = 5, pady = 5)
                    Button(current_layer_frame, text = "Delete layer", command = lambda i = layer_index - 1 : delete_layer(i)).grid(row = 1, column = 1, padx = 5, pady = 5)
                    Button(current_layer_frame, text = "Update layer", command = lambda i = layer_index - 1 : update_layer(i)).grid(row = 1, column = 2, padx = 5, pady = 5, sticky = 'e')
                    Button(internal_layers_frame, text = "Add layer", command = lambda i = layer_index : add_layer(i)).grid(row = (2 * layer_index + 1), column = 1, padx = 5, pady = 5, sticky = 'e')
                    internal_layers_frame.columnconfigure(0, weight=1)
                    internal_layers_frame.columnconfigure(1, weight=1)
                    internal_layers_frame.columnconfigure(2, weight=1)
                
                current_layer_frame = LabelFrame(internal_layers_frame)
                current_layer_frame.grid(row = (2 * MAX_LAYER_NUMBER), column = 0, columnspan = 3, padx = 5, pady = 4, sticky = 'w' + 'e')
                Label(current_layer_frame, text = f"{model['layers'][-1]} neuron(s)").grid(row = 1, column = 0, padx = 5, pady = 4, sticky = 'w')
                Label(internal_layers_frame, text = f"Input layer").grid(row = (2 * MAX_LAYER_NUMBER), column = 0, padx = 5, pady = 0, sticky = 'nw')
                
                Button(internal_layers_frame, text = "Cancel", command = cancel_changes).grid(row = (2 * MAX_LAYER_NUMBER + 2), column = 0, padx = 5, pady = 5)
                Button(internal_layers_frame, text = "Empty hidden layers", command = empty_hidden_layers).grid(row = (2 * MAX_LAYER_NUMBER + 2), column = 1, padx = 5, pady = 5)
                Button(internal_layers_frame, text = "Validate", command = layer_edit_window.destroy).grid(row = (2 * MAX_LAYER_NUMBER + 2), column = 2, padx = 5, pady = 5)
            #V

            nonlocal model
            previous_layers = model["layers"].copy()

            layer_edit_window = Tk()
            layer_edit_window.title("Fenetre de choix des consignes de l'essai")
            layer_edit_window.protocol("WM_delete_window", cancel_changes)
            layer_edit_window.rowconfigure(0, weight=1)
            layer_edit_window.columnconfigure(0, weight=1)
            layer_edit_window.columnconfigure(1, weight=0)
            
            canevas = Canvas(layer_edit_window, width = 900, height = 500)
            canevas.grid(row = 0, column = 0, sticky = (N, S, E, W))
            canevas.rowconfigure(0, weight=1)
            canevas.columnconfigure(0, weight=1)
            y_scrollbar = ttk.Scrollbar(layer_edit_window, orient="vertical", command=canevas.yview)
            y_scrollbar.grid(column=1, row=0, sticky=(N, S, E))
            internal_layers_frame = ttk.Frame(canevas, width = 880)
            internal_layers_frame.pack(in_ = canevas, expand = True, fill = BOTH)
            
            internal_layers_frame.bind("<Configure>", lambda _: canevas.configure(scrollregion = canevas.bbox("all")))
            canevas.create_window((0, 0), window = internal_layers_frame, anchor = "nw")
            canevas.configure(yscrollcommand=y_scrollbar.set)

            internal_layers_frame.bind('<Enter>', lambda e : _bound_to_mousewheel(canevas, e))
            internal_layers_frame.bind('<Leave>', lambda e : _unbound_to_mousewheel(canevas, e))

            actualisation_des_boutons()
            layer_edit_window.mainloop()
        #V
        
        def display_model_specs():
            # Is okay
            """ """
            nonlocal model
            for widget in model_type_frame.winfo_children()[4:] :
                widget.destroy()
            match model_type.get() :
                case "Linear" :
                    #TODO : check how I did the dic-filling window
                    Label(model_type_frame, text = f'{model["layers"][0]} input(s), {model["layers"][-1]} output(s)').grid(row = 2, column = 1, padx = 10, pady = 10)
                case "MLP" :
                    Label(model_type_frame, text = f'{model["layers"][0]} input(s), {len(model["layers"]) - 2} hidden layers, {model["layers"][-1]} output(s)').grid(row = 2, column = 1, padx = 10, pady = 10)
                    Button(model_type_frame, text = "Edit layers", command = edit_layers).grid(row = 2, column = 2, padx = 10, pady = 10)
                case "SVM" :
                    Label(model_type_frame, text = f'{model["layers"][0]} input(s), {len(model["layers"]) - 2} hidden layers, {model["layers"][-1]} output(s)').grid(row = 2, column = 1, padx = 10, pady = 10)
                    Button(model_type_frame, text = "Edit layers", command = edit_layers).grid(row = 2, column = 2, padx = 10, pady = 10)
                case "RBF" :
                    Label(model_type_frame, text = f'{model["layers"][0]} input(s), {len(model["layers"]) - 2} hidden layers, {model["layers"][-1]} output(s)').grid(row = 2, column = 1, padx = 10, pady = 10)
                    Button(model_type_frame, text = "Edit layers", command = edit_layers).grid(row = 2, column = 2, padx = 10, pady = 10)
                case _ :
                    pass
        #PV   facultatif

        def validate_entries():
            #TODO : save entries
            entries_window.destroy()
        #N
        
        def load_model_to_edit_from_file():
            """FR : Ouvre une fenêtre de sélection de fichier pour choisir une liste de consignes à charger.
            
            EN : Opens a file selection window to choose a list of setpoints to load."""
            load_model_from_file()
        #N
        
        def save_edited_model_as_file():
            """FR : Ouvre une fenêtre de sélection de fichier pour choisir où enregistrer la liste de consignes actuelle.
            
            EN : Opens a file selection window to choose where to save the current list of setpoints."""
            showwarning("The model has been modified. No loss-related data will be saved.")
            filepath = asksaveasfilename(initialdir = MODELS_SAVE_FOLDER, filetypes = [("fichier json", "*.json")], defaultextension = ".json")
            if filepath :
                with open(filepath, 'w') as model_file :
                    new_model = model.copy()
                    new_model["error_and_accuracy"] = []
                    dump(new_model, model_file)
        #V
        
        def compute_number_of_inputs():
            nonlocal model
            if inputs_width.get() and inputs_height.get():
                model["layers"][0] = int(inputs_width.get()) * int(inputs_height.get()) * inputs_color.get()
            display_model_specs()
        #V

        def set_last_layer():
            nonlocal model
            if classes_number.get() :
                model["layers"][-1] = int(classes_number.get())
            display_model_specs()
        #V
        
        def _check_entry_inputs(new_value):
            if _check_entry_unsigned_int(new_value):
                entries_window.after(1,compute_number_of_inputs)
                return True
            return False
        #V
        
        def _check_entry_classes(new_value):
            if _check_entry_unsigned_int(new_value):
                entries_window.after(1,set_last_layer)
                return True
            return False
        #V

        nonlocal model
        
        entries_window = Tk()
        entries_window.title("Configuration initiale")
        entries_window.protocol("WM_DELETE_WINDOW", exit)
        print(model)
        model_type = StringVar()
        model_type.set(str(model["model_type"]))
        print(model_type.get() + "|")
        print(model["model_type"] + "|")
        is_classification = BooleanVar()
        is_classification.set(bool(model["is_classification"]))
        print(is_classification.get())
        inputs_width = StringVar()
        inputs_width.set(model["inputs_width"])
        print(inputs_width.get())
        inputs_height = StringVar()
        inputs_height.set(model["inputs_height"])
        print(inputs_height.get())
        inputs_color = IntVar()
        inputs_color.set(model["inputs_color"])
        print(inputs_color.get())
        classes_number = StringVar()
        classes_number.set(model["layers"][-1])
        print(classes_number.get())
        # layers = model["layers"].copy()
        
        model_type_frame = LabelFrame(entries_window, text = "Model type :")
        model_type_frame.grid(row = 1, columnspan = 6, column = 0, padx =10, pady =10)
        Radiobutton(model_type_frame, text="Linear", variable=model_type, value="Linear", command = display_model_specs).grid(row = 0, column = 0, padx = 5, pady = 5)
        Radiobutton(model_type_frame, text="Multi-layer perceptron", variable=model_type, value="MLP", command = display_model_specs).grid(row = 0, column = 1,padx = 5, pady = 5)
        Radiobutton(model_type_frame, text="Support vector machine", variable=model_type, value="SVM", command = display_model_specs).grid(row = 0, column = 2,padx = 5, pady = 5)
        Radiobutton(model_type_frame, text="Radial basis function", variable=model_type, value="RBF", command = display_model_specs).grid(row = 0, column = 3,padx = 5, pady = 5)
        
        Label( entries_window, text = "Task type :").grid(row = 2, column = 0, padx =10, pady =10)
        Radiobutton(entries_window, text="Classification", variable=is_classification, value=True).grid(row = 2, column = 1, columnspan = 3, padx = 5, pady = 5)
        Radiobutton(entries_window, text="Regression", variable=is_classification, value=False).grid(row = 2, column = 5, columnspan = 2, padx = 5, pady = 5)
        
        Label(entries_window, text = "Input image size :").grid(row = 3, column = 0, padx =10, pady =10)
        Entry(entries_window, textvariable=inputs_width, width=5, validate="key", validatecommand=(entries_window.register(_check_entry_inputs), '%P')).grid(row = 3, column = 1, padx = 0, pady = 5)
        Label(entries_window, text = " x ").grid(row = 3, column = 2, padx = 5, pady = 0)
        Entry(entries_window, textvariable=inputs_height, width=5, validate="key", validatecommand=(entries_window.register(_check_entry_inputs), '%P')).grid(row = 3, column = 3, padx = 0, pady = 5)
        Radiobutton(entries_window, text="Grey", variable=inputs_color, value=1, command=compute_number_of_inputs).grid(row = 3, column = 4,padx = 5, pady = 5)
        Radiobutton(entries_window, text="RGB", variable=inputs_color, value=3, command=compute_number_of_inputs).grid(row = 3, column = 5,padx = 5, pady = 5)
        
        Label( entries_window, text = "Number of classes :").grid(row = 4, column = 0, padx = 10, pady = 10)
        Entry(entries_window, textvariable=classes_number, width=5, validate="key", validatecommand=(entries_window.register(_check_entry_classes), '%P')).grid(row = 4, column = 1, padx = 5, pady = 5)
        
        # Label(entries_window, text = "Titre").grid(row=1,column=0,padx =5, pady =5)
        # entree_titre=Entry(entries_window, textvariable=titre, width=30, validate="key", validatecommand=(entries_window.register(_check_entry_string), '%P'))
        # entree_titre.grid(row=1,c

        
        Button(entries_window, text="Cancel", command=entries_window.destroy).grid(row=20, column=0,padx =5, pady =5)
        Button(entries_window, text="Load model", command = load_model_to_edit_from_file).grid(row=20, column=1, columnspan = 2, padx =5, pady =5)
        Button(entries_window, text="Save model", command = save_edited_model_as_file).grid(row=20, column=3, columnspan = 2, padx =5, pady =5)
        Button(entries_window, text="Ok", command = validate_entries).grid(row=20, column=5 , padx =5, pady =5)

        menubar = Menu(entries_window)
        entries_window.config(menu=menubar)
        menu= Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Autre", menu=menu)
        menu.add_command(label="Afficher la documentation",command=RTM_protocol)

        entries_window.mainloop()
        
        return ()

    
    def enregistrement_des_documents_choisis():
        return

    def enregistrer_et_quitter():
        """FR : Fenêtre de choix des données à enregistrer avant de quitter ou relancer.

        EN : Window to choose which datas to save before quitting or launching another test.
        """

        def quitter():
            """FR : Enregistre et quitte.

            EN : Saves and quits."""
            choix_des_documents_a_enregistrer.set(
                check_val_brutes.get() + check_val_reelles.get()
            )
            if not sauvegarde_effectue:
                enregistrement_des_documents_choisis()
            # crappy_launch_thread._stop()
            exit()

        
        def relancer_un_essai():
            """FR : Enregistre et renvoie l'utilisateur sur la fenêtre de saisie des
               conditions du test.

            EN : Saves and brings back the user to the test conditions' entries window.
            """
            choix_des_documents_a_enregistrer.set(
                check_val_brutes.get() + check_val_reelles.get()
            )
            if not sauvegarde_effectue:
                enregistrement_des_documents_choisis()
            fenetre_de_sortie_du_programme.destroy()
            fenetre_principale.destroy()

        
        fenetre_de_sortie_du_programme = Toplevel(fenetre_principale)
        fenetre_de_sortie_du_programme.lift()

        check_val_brutes = IntVar()
        check_val_brutes.set(2 if choix_des_documents_a_enregistrer.get() >= 2 else 0)
        check_val_reelles = IntVar()
        check_val_reelles.set(
            1 if choix_des_documents_a_enregistrer.get() % 2 == 1 else 0
        )
        Label(
            fenetre_de_sortie_du_programme,
            text="Veuillez choisir les valeurs à conserver :",
        ).grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        ttk.Checkbutton(
            fenetre_de_sortie_du_programme,
            text="Valeurs réelles",
            variable=check_val_reelles,
            onvalue=1,
            offvalue=0,
        ).grid(row=2, column=0, columnspan=3, padx=10, pady=10)
        ttk.Checkbutton(
            fenetre_de_sortie_du_programme,
            text="Valeurs non étalonnées",
            variable=check_val_brutes,
            onvalue=2,
            offvalue=0,
        ).grid(row=3, column=0, columnspan=3, padx=10, pady=10)
        Button(
            fenetre_de_sortie_du_programme,
            text="Annuler",
            command=fenetre_de_sortie_du_programme.destroy,
        ).grid(row=5, column=0, padx=10, pady=10)
        Button(
            fenetre_de_sortie_du_programme,
            text="Relancer un essai",
            command=relancer_un_essai,
        ).grid(row=5, column=1, padx=10, pady=10)
        Button(fenetre_de_sortie_du_programme, text="Quitter", command=quitter).grid(
            row=5, column=2, padx=10, pady=10
        )

    def gros_bouton_rouge():
        exit()

    def choix_des_documents_a_conserver():
        """FR : Fenêtre de choix des documents à conserver.

        EN : Files to keep choice window."""

        def annulation():
            """FR : Restore le choix précédent et ferme cette fenêtre.

            EN : Sets back the previous choice and closes this window."""
            choix_des_documents_a_enregistrer.set(choix_actuel)
            fenetre_de_choix_des_doc_a_conserver.destroy()

        # V
        choix_actuel = choix_des_documents_a_enregistrer.get()
        fenetre_de_choix_des_doc_a_conserver = Toplevel(fenetre_principale)
        fenetre_de_choix_des_doc_a_conserver.protocol("WM_DELETE_WINDOW", annulation)

        Label(
            fenetre_de_choix_des_doc_a_conserver,
            text="Veuillez choisir les documents à enregistrer",
        ).grid(row=0, column=1, padx=10, pady=10)
        Radiobutton(
            fenetre_de_choix_des_doc_a_conserver,
            text="aucun document",
            variable=choix_des_documents_a_enregistrer,
            value=0,
        ).grid(row=1, column=1, padx=10, pady=10)
        Radiobutton(
            fenetre_de_choix_des_doc_a_conserver,
            text="que les trucs bof",
            variable=choix_des_documents_a_enregistrer,
            value=1,
        ).grid(row=2, column=1, padx=10, pady=10)
        Radiobutton(
            fenetre_de_choix_des_doc_a_conserver,
            text="que les trucs bien",
            variable=choix_des_documents_a_enregistrer,
            value=2,
        ).grid(row=3, column=1, padx=10, pady=10)
        Radiobutton(
            fenetre_de_choix_des_doc_a_conserver,
            text="TOUT ! JE VEUX TOUT !! MOUAHAHAHAHAH !!!",
            variable=choix_des_documents_a_enregistrer,
            value=3,
        ).grid(row=4, column=1, padx=10, pady=10)
        Button(
            fenetre_de_choix_des_doc_a_conserver, text="Retour", command=annulation
        ).grid(row=5, column=0, padx=10, pady=10)
        Button(
            fenetre_de_choix_des_doc_a_conserver,
            text="Suivant",
            command=fenetre_de_choix_des_doc_a_conserver.destroy,
        ).grid(row=5, column=2, padx=10, pady=10)
        fenetre_de_choix_des_doc_a_conserver.wait_window()

    # The following is a bunch of function that will be called upon pressing the UI's buttons.
    # Those are just skeletons, as of now. For each, we must :
    # - create adequate variables
    # - call the rust function
    # - delete the pointers if not needed anymore
    # - show some stuff, but not necessarily return anything

    def generate_linear_model():
        pass


    def train_linear_model():
        pass
    
    
    def test_linear_model():
        pass


    def save_linear_model():
        pass


    def load_linear_model():
        pass


    def generate_mlp():
        pass


    def train_mlp():
        pass
    
    
    def test_mlp():
        pass


    def save_mlp():
        pass
    
    
    def load_mlp():
        pass


    
    model = {"model_type" : "Linear",
             "layers" : [1, 1],
             "error_and_accuracy" : [],
             "inputs_width" : 1,
             "inputs_height" : 1,
             "inputs_color" : 1,
             "dataset_folders" : [""],
             "is_classification" : True}
    
    # each element in error_and_accuracy is a dict looking like :
    # {
    # "number_of_train_inputs" : x,
    # "number_of_test_inputs" : x,
    # "number_of_classes": x,
    # "number_of_epochs" : x,
    # "batch_size" : x,
    # "number_of_errors" : [],
    # "accurracy" : []
    # }
    
    





    fenetre_principale = Tk()
    fenetre_principale.title("C'est la classe ! Mais laquelle ?")
    fenetre_principale.protocol("WM_DELETE_WINDOW", enregistrer_et_quitter)

    # Variables nécessaires à l'enregistrement

    choix_des_documents_a_enregistrer = IntVar()
    sauvegarde_effectue = False

    ##### Organisation de l'affichage #####
    liste_des_ecrans = get_monitors()
    indice_ecran = 0
    while (
        indice_ecran < len(liste_des_ecrans)
        and not liste_des_ecrans[indice_ecran].is_primary
    ):
        indice_ecran += 1  # Cherche l'écran principal.
    largeur_de_l_ecran = liste_des_ecrans[indice_ecran].width
    hauteur_de_l_ecran = liste_des_ecrans[indice_ecran].height
    # largeur_de_l_ecran = 1440
    canvas = Canvas(
        fenetre_principale,
        height=int(hauteur_de_l_ecran / 2),
        width=largeur_de_l_ecran / 3,
    )
    canvas.grid(column=0, row=0, columnspan=1, sticky=(N, W, E, S))
    width_scrollbar = ttk.Scrollbar(fenetre_principale, orient = HORIZONTAL, command=canvas.xview)
    width_scrollbar.grid(column=0, row=1, sticky=(W, E))
    height_scrollbar = ttk.Scrollbar(fenetre_principale, orient = VERTICAL, command=canvas.yview)
    height_scrollbar.grid(column=1, row=0, sticky=(N, S))
    cadre_interne = Frame(canvas)
    canvas.configure(xscrollcommand = width_scrollbar.set)
    canvas.configure(yscrollcommand = height_scrollbar.set)
    cadre_interne.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=cadre_interne, anchor="nw")

    cadre_interne.bind("<Enter>", lambda e : _bound_to_mousewheel(canvas, e))
    cadre_interne.bind("<Leave>", lambda e : _unbound_to_mousewheel(canvas, e))

    # Quand on modifie la taille de la fenêtre, la scrollbar reste de la même taille et
    # le reste s'agrandit.
    fenetre_principale.rowconfigure(0, weight=1)
    fenetre_principale.rowconfigure(1, weight=0)
    fenetre_principale.columnconfigure(0, weight=1)
    fenetre_principale.columnconfigure(1, weight=0)
    canvas.rowconfigure(0, weight=1)
    canvas.columnconfigure(0, weight=1)

    bouton_de_demarrage_du_test = Button(cadre_interne, text="Train MLP")
    bouton_de_lancement_de_l_enregistrement = Button(cadre_interne, text="Edit model", command = model_configuration)
    bouton_de_mise_en_tension_rapide = Button(cadre_interne, text="Test MLP")
    # bouton_de_mise_en_tension_lente=Button(cadre_interne, text="Mise en tension lente")#, command = mise_en_tension_lente)
    # bouton_de_retour_en_position_initiale=Button(cadre_interne, text="Retour en position 0")#, command = retour_en_position_initiale)
    bouton_d_arret_de_crappy = Button(cadre_interne, text="Pause", command=gros_bouton_rouge)  # ,bg='red')
    bouton_enregistrer_et_quitter = Button(cadre_interne, text="Quitter et enregistrer", command=enregistrer_et_quitter)

    bouton_enregistrer = Button(cadre_interne, text=" ", command=choix_des_documents_a_conserver)

    img1 = PhotoImage(file=CONFIG_FOLDER + "icone_enregistrer.png")
    bouton_enregistrer.config(image=img1)
    # img2 = PhotoImage(file= CONFIG_FOLDER + "icone_engrenage.png")
    # bouton_parametrage_consigne.config(image=img2)
    # img3 = PhotoImage(file= CONFIG_FOLDER + "icone_retour.png")
    # mise_a_0_btn.config(image=img3)
    img6 = PhotoImage(file=CONFIG_FOLDER + "icone_play.png")
    bouton_de_demarrage_du_test.config(image=img6)
    img7 = PhotoImage(file=CONFIG_FOLDER + "loss.png")
    bouton_d_arret_de_crappy.config(image=img7)
    # img8 = PhotoImage(file= CONFIG_FOLDER + "icone_tension.png")

    bouton_de_demarrage_du_test.grid(row=0, column=13, padx=5, pady=5)
    bouton_de_lancement_de_l_enregistrement.grid(row=1, column=13, padx=5, pady=5)
    bouton_de_mise_en_tension_rapide.grid(row=2, column=13, padx=5, pady=5)
    # bouton_de_mise_en_tension_lente.grid(row=3,column=13,padx =5, pady =5)
    # bouton_de_retour_en_position_initiale.grid(row=4,column=13,padx =5, pady =5)
    bouton_d_arret_de_crappy.grid(row=0, column=14, columnspan=2, padx=5, pady=5)
    bouton_enregistrer_et_quitter.grid(row=2, column=14, padx=5, pady=5)

    bouton_enregistrer.grid(row=1, column=14, padx=5, pady=5)

    fenetre_principale.mainloop()

    return fonction_principale()


if __name__ == "__main__":
    fonction_principale()
