#!/usr/bin/env python
# -*- encoding: utf-8 -*-

### Opérations sur les fichiers
import os
from os.path import normpath
from PIL import Image

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
def _check_entry_int(new_value):
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


def read_config_data(file_name):
    """FR : Renvoie la dernière entrée du fichier texte indiqué. Les fichiers utilisant
    cette fonction ne doivent être modifiés que par les fonctions de cette application.

    EN : Returns the last entry in the indicated text file. Files using this function musn't
    be modified except by functions of this application."""
    with open(normpath(file_name), "r") as f:
        lines = f.readlines()
        data = lines[-1][11:-1]  # Datas are formated as "yyyy-mm-dd <data>\n".
    return data


# Chemins des fichiers
CONFIG_FOLDER = read_config_data("UI/dossier_config_et_consignes.txt") + "/"
SAVE_FOLDER = read_config_data("UI/dossier_enregistrements.txt") + "/"
# print (SAVE_FOLDER)


def RTM_protocol():
   """FR : Ouvre le manuel de la bibliothèque.

   EN : Opens the library's manual."""
   os.startfile(read_config_data(r"./README.md"))
#V

def model_configuration (init_model_type: str = "MLP",
                         init_inputs_width: int = 1,
                         init_inputs_height: int = 1,
                         init_inputs_color: int = 1,
                         init_dataset_folders: List[str] = [], # ptet à mettre ailleurs
                         init_layers: List[int] = [1, 1],
                         init_is_classification: bool = True) :
    """FR : Fenêtre de configuration des valeurs initiales de l'essai.
    
    EN : Test's initial values configuration window."""

    def edit_layers():
        """FR : Fenêtre de configuration des consignes à suivre pendant l'essai.
        
        EN : Setup window for the setpoints to follow during the test."""
        def surcouche_ajout(new_layer_index):
            """FR : Évite de rajouter une consigne si l'utilisateur annule.
            
            EN : Avoids adding a setpoint if the user cancels."""
            consigne_a_modifier = ajout_ou_modification_d_une_consigne()
                    
            if consigne_a_modifier is not None :
                layers.append(consigne_a_modifier)
                for i in range (len(layers)-1, new_layer_index, -1) :
                    layers[i], layers[i - 1] = layers[i - 1], layers[i]
                return actualisation_des_boutons()
            
        def surcouche_modification(updating_layer_index):
            """FR : Évite de modifier la consigne si l'utilisateur annule.
            
            EN : Avoids modifying the setpoint if the user cancels."""
            nonlocal layers
            consigne_a_modifier = ajout_ou_modification_d_une_consigne(layers[updating_layer_index])
            if consigne_a_modifier is not None :
                layers[updating_layer_index] = consigne_a_modifier
                return actualisation_des_boutons()
        #V
        def ajout_ou_modification_d_une_consigne(consigne_a_changer: int = 0):
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
            layer_edit_window = Toplevel(fenetre_de_choix_des_consignes)
            if is_an_update :
                layer_edit_window.title("Update layer")
            else :
                layer_edit_window.title("Add layer")
            
            neurons_number = IntVar()
            if is_an_update :
                neurons_number.set(consigne_a_changer)
            Label(layer_edit_window, text = "Neurons number :").grid(row = 1, column = 0, padx = 5, pady = 5)
            Entry(layer_edit_window, width = 5, textvariable = neurons_number, validate="key", validatecommand=(layer_edit_window.register(_check_entry_int), '%P')).grid(row = 1, column = 1, padx = 5, pady = 5)

            Button(layer_edit_window, text = "Annuler", command = layer_edit_window.destroy).grid(row = 100, column = 0, columnspan = 2, padx = 5, pady = 5)
            Button(layer_edit_window, text = "Valider", command = ajout_ou_modification_validee).grid(row = 100, column = 2, columnspan = 3, padx = 5, pady = 5)
            
            layer_edit_window.wait_window()
            if validation :
                return neurons_number.get()
        #V
        def suppression_de_toutes_les_consignes():
            """FR : Supprime toutes les consignes.
            
            EN : Deletes all the setpoints."""
            nonlocal layers
            layers = []
            return actualisation_des_boutons()
        #N
        def annulation_des_changements():
            """FR : Annule les changements et restore la lsite de consignes précédente.
            
            EN : Cancels the changes and restores the list to its former state."""
            nonlocal layers
            layers = consignes_precedentes
            fenetre_de_choix_des_consignes.destroy()
        #N
        def suppression_d_une_consigne(indice_consigne):
            """FR : Supprime la consigne donnée.
            
            EN : Deletes the given setpoint."""
            nonlocal layers
            layers.pop(indice_consigne)
            return actualisation_des_boutons()
        #N
        def chargement_des_consignes():
            """FR : Ouvre une fenêtre de sélection de fichier pour choisir une liste de consignes à charger.
            
            EN : Opens a file selection window to choose a list of setpoints to load."""
            chemin_du_fichier = askopenfilename(title = "Consignes à charger", initialdir = CONFIG_FOLDER, filetypes = [("fichier json", "*.json")])
            if chemin_du_fichier :
                with open(chemin_du_fichier, 'r') as fichier_de_consignes :
                    nonlocal layers
                    consignes_potentielles = load(fichier_de_consignes)
                    for i in range(len(consignes_potentielles)) :
                        if "type" not in consignes_potentielles[i].keys():
                            showwarning("Fichier incorrect : aucune consigne trouvée.")
                            break
                        elif i == len(consignes_potentielles) - 1 :
                            layers = consignes_potentielles
                actualisation_des_boutons()
        #N
        def enregistrement_des_consignes():
            """FR : Ouvre une fenêtre de sélection de fichier pour choisir où enregistrer la liste de consignes actuelle.
            
            EN : Opens a file selection window to choose where to save the current list of setpoints."""
            chemin_du_fichier = asksaveasfilename(initialdir = CONFIG_FOLDER, filetypes = [("fichier json", "*.json")], defaultextension = ".json")
            if chemin_du_fichier :
                with open(chemin_du_fichier, 'w') as fichier_de_consignes :
                    dump(layers, fichier_de_consignes)
        #N
        def actualisation_des_boutons():
            """FR : Gère l'affichage des consignes et de leurs boutons associés.
            
            EN : Manages the display of the setpoints and their associated buttons."""
            for widget in cadre_interne_consignes.winfo_children() :
                widget.destroy()

            if verrou_production == OFF :
                Button(cadre_interne_consignes, text = "Charger depuis un fichier", command = chargement_des_consignes).grid(row = 0, column = 0, padx = 5, pady = 12)
                Button(cadre_interne_consignes, text = "Insérer une consigne au départ", command = lambda : surcouche_ajout(0)).grid(row = 0, column = 1, padx = 5, pady = 12)
                Button(cadre_interne_consignes, text = "Enregistrer dans un fichier", command = enregistrement_des_consignes).grid(row = 0, column = 2, padx = 5, pady = 12)
                if len(layers) :
                    Label(cadre_interne_consignes, text = "Consigne(s) actuellement prévue(s) :").grid(row = 1, column = 0, columnspan = 3, padx = 5, pady = 4)
                    layer_index = 0
                    for consigne_du_generateur in layers :
                        layer_index += 1
                        label_de_cette_consigne = ""
                        if type_d_asservissement == ASSERVISSEMENT_EN_CHARGE :
                            match consigne_du_generateur['type'] :
                                case "ramp" :
                                    label_de_cette_consigne = f"Rampe simple de {2 * consigne_du_generateur['speed']}T/s"
                                    condition_d_arret = consigne_du_generateur["condition"]
                                    if condition_d_arret is None :
                                        label_de_cette_consigne += ", dure indéfiniment"
                                    elif condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" jusqu'à {str(2 * float(condition_d_arret[DEBUT_CONDITION_CHARGE:]))}T"
                                case "constant" :
                                    label_de_cette_consigne = "Palier à "
                                    label_de_cette_consigne += f"{2 * consigne_du_generateur['value']}T"
                                    condition_d_arret = consigne_du_generateur["condition"]
                                    if condition_d_arret is None :
                                        label_de_cette_consigne += ", dure indéfiniment"
                                    elif condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" maintenu jusqu'à atteindre {condition_d_arret[DEBUT_CONDITION_CHARGE:]}T"
                                case "cyclic_ramp" :
                                    label_de_cette_consigne = f"{consigne_du_generateur['cycles']} cycles de rampes : "
                                    label_de_cette_consigne += f"{2 * consigne_du_generateur['speed1']}T/s"
                                    condition_d_arret = consigne_du_generateur["condition1"]
                                    if condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" jusqu'à {str(2 * float(condition_d_arret[DEBUT_CONDITION_CHARGE:]))}T"
                                    label_de_cette_consigne += f", {2 * consigne_du_generateur['speed2']}T/s"
                                    condition_d_arret = consigne_du_generateur["condition2"]
                                    if condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" jusqu'à {str(2 * float(condition_d_arret[DEBUT_CONDITION_CHARGE:]))}T"
                                case "cyclic" :
                                    label_de_cette_consigne = f"{consigne_du_generateur['cycles']} cycles de paliers : "
                                    label_de_cette_consigne += f"{2 * consigne_du_generateur['value1']}T"
                                    condition_d_arret = consigne_du_generateur["condition1"]
                                    if condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" jusqu'à {condition_d_arret[DEBUT_CONDITION_CHARGE:]}T"
                                    label_de_cette_consigne += f", {2 * consigne_du_generateur['value2']}T"
                                    condition_d_arret = consigne_du_generateur["condition2"]
                                    if condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" jusqu'à {condition_d_arret[DEBUT_CONDITION_CHARGE:]}T"
                                case "sine" :
                                    label_de_cette_consigne = f"Sinus allant de {2 * (consigne_du_generateur['offset'] - consigne_du_generateur['amplitude'] / 2)}T à {2 * (consigne_du_generateur['offset'] + consigne_du_generateur['amplitude'] / 2)}T, de période {1 / consigne_du_generateur['freq']}s, démarrant "
                                    match int(consigne_du_generateur['phase'] * 2 / pi + 0.05) :
                                        case 0 :
                                            label_de_cette_consigne += "croissant au centre"
                                        case 1 :
                                            label_de_cette_consigne += "à son maximum"
                                        case 2 :
                                            label_de_cette_consigne += "décroissant au centre"
                                        case 3 :
                                            label_de_cette_consigne += "à son minimum"
                                    condition_d_arret = consigne_du_generateur["condition"]
                                    if condition_d_arret is None :
                                        label_de_cette_consigne += ", dure indéfiniment"
                                    else :
                                        label_de_cette_consigne += f" pendant {round(float(condition_d_arret[DEBUT_CONDITION_TEMPS:]) * consigne_du_generateur['freq'], 2)} cycles"
                        else :
                            match consigne_du_generateur['type'] :
                                case "ramp" :
                                    label_de_cette_consigne = f"Rampe simple de {COEF_VOLTS_TO_MILLIMETERS * consigne_du_generateur['speed']}mm/s"
                                    condition_d_arret = consigne_du_generateur["condition"]
                                    if condition_d_arret is None :
                                        label_de_cette_consigne += ", dure indéfiniment"
                                    elif condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" jusqu'à {str(COEF_VOLTS_TO_MILLIMETERS * float(condition_d_arret[DEBUT_CONDITION_POSITION:]))}mm"
                                case "constant" :
                                    label_de_cette_consigne = "Palier à "
                                    label_de_cette_consigne += f"{COEF_VOLTS_TO_MILLIMETERS * consigne_du_generateur['value']}mm"
                                    condition_d_arret = consigne_du_generateur["condition"]
                                    if condition_d_arret is None :
                                        label_de_cette_consigne += ", dure indéfiniment"
                                    elif condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" maintenu jusqu'à atteindre {COEF_VOLTS_TO_MILLIMETERS * condition_d_arret[DEBUT_CONDITION_POSITION:]}T"
                                case "cyclic_ramp" :
                                    label_de_cette_consigne = f"{consigne_du_generateur['cycles']} cycles de rampes : "
                                    label_de_cette_consigne += f"{COEF_VOLTS_TO_MILLIMETERS * consigne_du_generateur['speed1']}mm/s"
                                    condition_d_arret = consigne_du_generateur["condition1"]
                                    if condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" jusqu'à {str(COEF_VOLTS_TO_MILLIMETERS * float(condition_d_arret[DEBUT_CONDITION_POSITION:]))}T"
                                    label_de_cette_consigne += f", {COEF_VOLTS_TO_MILLIMETERS * consigne_du_generateur['speed2']}mm/s"
                                    condition_d_arret = consigne_du_generateur["condition2"]
                                    if condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" jusqu'à {str(COEF_VOLTS_TO_MILLIMETERS * float(condition_d_arret[DEBUT_CONDITION_POSITION:]))}T"
                                case "cyclic" :
                                    label_de_cette_consigne = f"{consigne_du_generateur['cycles']} cycles de paliers : "
                                    label_de_cette_consigne += f"{COEF_VOLTS_TO_MILLIMETERS * consigne_du_generateur['value1']}mm"
                                    condition_d_arret = consigne_du_generateur["condition1"]
                                    if condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" jusqu'à {COEF_VOLTS_TO_MILLIMETERS * condition_d_arret[DEBUT_CONDITION_POSITION:]}mm"
                                    label_de_cette_consigne += f", {COEF_VOLTS_TO_MILLIMETERS * consigne_du_generateur['value2']}mm"
                                    condition_d_arret = consigne_du_generateur["condition2"]
                                    if condition_d_arret.startswith('delay') :
                                        label_de_cette_consigne += f" pendant {condition_d_arret[DEBUT_CONDITION_TEMPS:]}s"
                                    else :
                                        label_de_cette_consigne += f" jusqu'à {COEF_VOLTS_TO_MILLIMETERS * condition_d_arret[DEBUT_CONDITION_POSITION:]}mm"
                                case "sine" :
                                    label_de_cette_consigne = f"Sinus allant de {COEF_VOLTS_TO_MILLIMETERS * (consigne_du_generateur['offset'] - consigne_du_generateur['amplitude'] / 2)}mm à {COEF_VOLTS_TO_MILLIMETERS * (consigne_du_generateur['offset'] + consigne_du_generateur['amplitude'] / 2)}mm, de période {1 / consigne_du_generateur['freq']}s, démarrant "
                                    match int(consigne_du_generateur['phase'] * 2 / pi + 0.05) :
                                        case 0 :
                                            label_de_cette_consigne += "croissant au centre"
                                        case 1 :
                                            label_de_cette_consigne += "à son maximum"
                                        case 2 :
                                            label_de_cette_consigne += "décroissant au centre"
                                        case 3 :
                                            label_de_cette_consigne += "à son minimum"
                                    condition_d_arret = consigne_du_generateur["condition"]
                                    if condition_d_arret is None :
                                        label_de_cette_consigne += ", dure indéfiniment"
                                    else :
                                        label_de_cette_consigne += f" pendant {round(float(condition_d_arret[DEBUT_CONDITION_TEMPS:]) * consigne_du_generateur['freq'], 2)} cycles"

                        cadre_de_cette_consigne = LabelFrame(cadre_interne_consignes)
                        cadre_de_cette_consigne.grid(row = (2 * layer_index), column = 0, columnspan = 3, padx = 5, pady = 4, sticky = 'w' + 'e')
                        Label(cadre_de_cette_consigne, text = label_de_cette_consigne).grid(row = 1, column = 0, padx = 5, pady = 4, sticky = 'w')
                        label_du_numero_de_bloc = Label(cadre_interne_consignes, text = f"Bloc {layer_index}")
                        label_du_numero_de_bloc.grid(row = (2 * layer_index), column = 0, padx = 5, pady = 0, sticky = 'nw')
                        # fenetre_de_choix_des_consignes.wm_attributes('-transparentcolor', label_du_numero_de_bloc['bg'])
                        # Label(cadre_de_cette_consigne, image = PhotoImage(file = CONFIG_FOLDER + "rampe simple.png")).grid(row = 1, column = 0, padx = 5, pady = 5)
                        Button(cadre_de_cette_consigne, text = "Supprimer cette consigne", command = lambda i = layer_index - 1 : suppression_d_une_consigne(i)).grid(row = 1, column = 1, padx = 5, pady = 5)
                        Button(cadre_de_cette_consigne, text = "Modifier cette consigne", command = lambda i = layer_index - 1 : surcouche_modification(i)).grid(row = 1, column = 2, padx = 5, pady = 5, sticky = 'e')
                        Button(cadre_interne_consignes, text = "Insérer une consigne", command = lambda i = layer_index : surcouche_ajout(i)).grid(row = (2 * layer_index + 1), column = 1, padx = 5, pady = 5, sticky = 'e')
                        cadre_interne_consignes.columnconfigure(0, weight=1)
                        cadre_interne_consignes.columnconfigure(1, weight=1)
                        cadre_interne_consignes.columnconfigure(2, weight=1)
            else :
                if len(layers) :
                    Label(cadre_interne_consignes, text = "Consigne actuellement prévue :").grid(row = 0, column = 0, columnspan = 3, padx = 5, pady = 4)
                    label_de_cette_consigne = f"Tire le cable"# à {2 * layers[2]['speed']}T/s"
                    label_de_cette_consigne += f" jusqu'à {str(2 * float(layers[2]['condition'][DEBUT_CONDITION_CHARGE:]))}T"
                    Label(cadre_interne_consignes, text = label_de_cette_consigne).grid(row = 1, column = 0, columnspan = 3, padx = 5, pady = 4, sticky = 'we')

            Button(cadre_interne_consignes, text = "Annuler les changements", command = annulation_des_changements).grid(row = (2 * NOMBRE_DE_CONSIGNES_MAXIMAL + 2), column = 0, padx = 5, pady = 5)
            if verrou_production == OFF :
                Button(cadre_interne_consignes, text = "Tout supprimer", command = suppression_de_toutes_les_consignes).grid(row = (2 * NOMBRE_DE_CONSIGNES_MAXIMAL + 2), column = 1, padx = 5, pady = 5)
            else :
                Button(cadre_interne_consignes, text = "Modifier", command = lambda : surcouche_modification(2)).grid(row = (2 * NOMBRE_DE_CONSIGNES_MAXIMAL + 2), column = 1, padx = 5, pady = 5)
            Button(cadre_interne_consignes, text = "Valider", command = fenetre_de_choix_des_consignes.destroy).grid(row = (2 * NOMBRE_DE_CONSIGNES_MAXIMAL + 2), column = 2, padx = 5, pady = 5)
        #N

        nonlocal layers
        consignes_precedentes = layers.copy()

        fenetre_de_choix_des_consignes = Tk()
        fenetre_de_choix_des_consignes.title("Fenetre de choix des consignes de l'essai")
        fenetre_de_choix_des_consignes.protocol("WM_delete_window", annulation_des_changements)
        fenetre_de_choix_des_consignes.rowconfigure(0, weight=1)
        fenetre_de_choix_des_consignes.columnconfigure(0, weight=1)
        fenetre_de_choix_des_consignes.columnconfigure(1, weight=0)
        if verrou_production == OFF :
            canevas = Canvas(fenetre_de_choix_des_consignes, width = 900, height = 500)
            canevas.grid(row = 0, column = 0, sticky = (N, S, E, W))
            canevas.rowconfigure(0, weight=1)
            canevas.columnconfigure(0, weight=1)
            y_scrollbar = ttk.Scrollbar(fenetre_de_choix_des_consignes, orient="vertical", command=canevas.yview)
            y_scrollbar.grid(column=1, row=0, sticky=(N, S, E))
            cadre_interne_consignes = ttk.Frame(canevas, width = 880)
            cadre_interne_consignes.pack(in_ = canevas, expand = True, fill = BOTH)
            
            cadre_interne_consignes.bind("<Configure>", lambda _: canevas.configure(scrollregion = canevas.bbox("all")))
            canevas.create_window((0, 0), window = cadre_interne_consignes, anchor = "nw")
            canevas.configure(yscrollcommand=y_scrollbar.set)

            cadre_interne_consignes.bind('<Enter>', lambda e : _bound_to_mousewheel(canevas, e))
            cadre_interne_consignes.bind('<Leave>', lambda e : _unbound_to_mousewheel(canevas, e))
        else :
            cadre_interne_consignes = ttk.Frame(fenetre_de_choix_des_consignes)
            cadre_interne_consignes.pack(expand = True)

        actualisation_des_boutons()
        fenetre_de_choix_des_consignes.mainloop()
    #N

    def validate_entries():
        #TODO : save entries
        entries_window.destroy()
    #PV   facultatif
    
    def compute_number_of_inputs():
        nonlocal layers
        layers[0] = inputs_width.get() * inputs_height.get() * inputs_color.get()
    #V
    
    def display_model_specs():
        # Is okay
        """ """
        # match model_type.get() :
        #     case 0 :
        if model_type.get() == 0 :
                #TODO : check how I did the dic-filling window
                Label(model_type_frame, text = "WIP").grid(row = 0, column = 0, padx = 10, pady = 10)
                Button(model_type_frame, text = "Edit layers", command = edit_layers).grid(row = 1, column = 0, padx = 10, pady = 10)
            # case 2 :
            #     pass
            # case 3 :
            #     pass
            # case 4 :
            #     pass
            # case _ :
            #     break
        else :
            for widget in model_type_frame.winfo_children()[1:] :
                widget.destroy()
    #PV   facultatif
    global verrou_production

    entries_window = Tk()
    entries_window.title("Configuration initiale")
    entries_window.protocol("WM_DELETE_WINDOW", exit)

    model_type = StringVar()
    model_type.set(init_model_type)
    is_classification = BooleanVar()
    is_classification.set(init_model_type)
    inputs_width = IntVar()
    inputs_width.set(init_inputs_width)
    inputs_height = IntVar()
    inputs_height.set(init_inputs_height)
    inputs_color = IntVar()
    inputs_color.set(init_inputs_color)
    classes_number = IntVar()
    classes_number.set(init_layers[-1])
    layers = init_layers.copy()
    
    model_type_frame = LabelFrame(entries_window, text = "Model type :")
    model_type_frame.grid(row = 1, column = 0, padx =10, pady =10)
    Radiobutton(model_type_frame, text="Linear", variable=model_type, value="Linear", command = display_model_specs).grid(row = 0, column = 0, padx = 5, pady = 5)
    Radiobutton(model_type_frame, text="Multi-layer perceptron", variable=model_type, value="MLP", command = display_model_specs).grid(row = 0, column = 1,padx = 5, pady = 5)
    Radiobutton(model_type_frame, text="Support vector machine", variable=model_type, value="SVM", command = display_model_specs).grid(row = 0, column = 2,padx = 5, pady = 5)
    Radiobutton(model_type_frame, text="Radial basis function", variable=model_type, value="RBF", command = display_model_specs).grid(row = 0, column = 3,padx = 5, pady = 5)
    
    Label( entries_window, text = "Task type :").grid(row = 2, column = 0, padx =10, pady =10)
    Radiobutton(entries_window, text="Classification", variable=is_classification, value=True).grid(row = 2, column = 1, padx = 5, pady = 5)
    Radiobutton(entries_window, text="Regression", variable=is_classification, value=False).grid(row = 2, column = 2,padx = 5, pady = 5)
    
    Label(entries_window, text = "Input image size :").grid(row = 2, column = 0, padx =10, pady =10)
    Entry(entries_window, textvariable=inputs_width, width=5, command=compute_number_of_inputs, validate="key", validatecommand=(entries_window.register(_check_entry_int), '%P')).grid(row = 2, column = 1, padx = 5, pady = 5)
    Label(entries_window, text = " x ").grid(row = 2, column = 2, padx = 5, pady = 5)
    Entry(entries_window, textvariable=inputs_height, width=5, command=compute_number_of_inputs, validate="key", validatecommand=(entries_window.register(_check_entry_int), '%P')).grid(row = 2, column = 3, padx = 5, pady = 5)
    Radiobutton(entries_window, text="Grey", variable=inputs_color, value=1, command=compute_number_of_inputs).grid(row = 2, column = 4,padx = 5, pady = 5)
    Radiobutton(entries_window, text="RGB", variable=inputs_color, value=3, command=compute_number_of_inputs).grid(row = 2, column = 5,padx = 5, pady = 5)
    
    Label( entries_window, text = "Task type").grid(row = 2, column = 0, padx =10, pady =10)
    Entry(entries_window, textvariable=classes_number, variable=is_classification, value=True).grid(row = 2, column = 1, padx = 5, pady = 5)
    Radiobutton(entries_window, textvariable=classes_number, variable=is_classification, value=False).grid(row = 2, column = 2,padx = 5, pady = 5)
    
    # Label(entries_window, text = "Titre").grid(row=1,column=0,padx =5, pady =5)
    # entree_titre=Entry(entries_window, textvariable=titre, width=30, validate="key", validatecommand=(entries_window.register(_check_entry_string), '%P'))
    # entree_titre.grid(row=1,c

    
    Button(entries_window, text='Cancel', command=entries_window.destroy).grid(row=20, column=0,padx =5, pady =5)
    Button(entries_window, text='Ok', command = validate_entries).grid(row=20, column=1,padx =5, pady =5)

    menubar = Menu(entries_window)
    entries_window.config(menu=menubar)
    menu= Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Autre", menu=menu)
    menu.add_command(label="Afficher la documentation",command=RTM_protocol)

    entries_window.mainloop()
    
    return ()



def fonction_principale():
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


    
    entrees = False
    while entrees == False :
        entrees = model_configuration()
        type_d_asservissement = 1    





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
    width_scrollbar = ttk.Scrollbar(
        fenetre_principale, orient=HORIZONTAL, command=canvas.xview
    )
    width_scrollbar.grid(column=0, row=1, sticky=(W, E))
    cadre_interne = Frame(canvas)
    canvas.configure(xscrollcommand=width_scrollbar.set)
    cadre_interne.bind(
        "<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=cadre_interne, anchor="nw")

    cadre_interne.bind("<Enter>", lambda e: _bound_to_mousewheel(canvas, e))
    cadre_interne.bind("<Leave>", lambda e: _unbound_to_mousewheel(canvas, e))

    # Quand on modifie la taille de la fenêtre, la scrollbar reste de la même taille et
    # le reste s'agrandit.
    fenetre_principale.rowconfigure(0, weight=1)
    fenetre_principale.rowconfigure(1, weight=0)
    fenetre_principale.columnconfigure(0, weight=1)
    canvas.rowconfigure(0, weight=1)
    canvas.columnconfigure(0, weight=1)

    bouton_de_demarrage_du_test = Button(cadre_interne, text="Train MLP")
    bouton_de_lancement_de_l_enregistrement = Button(cadre_interne, text="Validate MLP")
    bouton_de_mise_en_tension_rapide = Button(cadre_interne, text="Test MLP")
    # bouton_de_mise_en_tension_lente=Button(cadre_interne, text="Mise en tension lente")#, command = mise_en_tension_lente)
    # bouton_de_retour_en_position_initiale=Button(cadre_interne, text="Retour en position 0")#, command = retour_en_position_initiale)
    bouton_d_arret_de_crappy = Button(
        cadre_interne, text="Pause", command=gros_bouton_rouge
    )  # ,bg='red')
    bouton_enregistrer_et_quitter = Button(
        cadre_interne, text="Quitter et enregistrer", command=enregistrer_et_quitter
    )

    bouton_enregistrer = Button(
        cadre_interne, text=" ", command=choix_des_documents_a_conserver
    )

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
