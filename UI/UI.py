#!/usr/bin/env python
# -*- encoding: utf-8 -*-

### Opérations sur les fichiers
from os.path import normpath

### Interface graphique
from tkinter import *
from tkinter import ttk

# from tkinter import tix # Obsolète, à remplacer
from tkinter.messagebox import *
from tkinter.filedialog import *
from screeninfo import get_monitors

### Divers
from numpy import pi
import time
import datetime
import re
from threading import Event, Thread

### Debug
import sys
import random


### Fonctions gérant le défilement
def _bound_to_mousewheel(widget, event):
    """FR : Lie le défilement de la fenêtre à la molette de la souris lorsque le curseur est sur
    cette fenêtre.

    EN : Binds the window's scrolling to the mousewheel when the cursor is over that window.
    """
    widget.bind_all("<MouseWheel>", lambda e: _on_mousewheel(widget, e))


# V
def _unbound_to_mousewheel(widget, event):
    """FR : Délie le défilement de la fenêtre à la molette de la souris lorsque le curseur sort
    de cette fenêtre.

    EN : Binds the window's scrolling to the mousewheel when the cursor leaves that window.
    """
    widget.unbind_all("<MouseWheel>")


# V
def _on_mousewheel(widget, event):
    """FR : Fait défiler la fenêtre avec la molette.

    EN : Scrolls the window with the mousewheel."""
    widget.yview_scroll(int(-1 * (event.delta / 80)), "units")


# V


def lecture_donnee(file_name):
    """FR : Renvoie la dernière entrée du fichier texte indiqué. Les fichiers utilisant
    cette fonction ne doivent être modifiés que par les fonctions de cette application.

    EN : Returns the last entry in the indicated text file. Files using this function musn't
    be modified except by functions of this application."""
    with open(normpath(file_name), "r") as f:
        lines = f.readlines()
        data = lines[-1][11:-1]  # Datas are formated as "yyyy-mm-dd <data>\n".
    return data


# V
# Chemins des fichiers
DOSSIER_CONFIG_ET_CONSIGNES = lecture_donnee("dossier_config_et_consignes.txt") + "/"
DOSSIER_ENREGISTREMENTS = lecture_donnee("dossier_enregistrements.txt") + "/"
# print (DOSSIER_ENREGISTREMENTS)


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

        # V
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

        # V
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

    fenetre_principale = Tk()
    fenetre_principale.title("wsh cé tro 1 bn prgrm")
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

    img1 = PhotoImage(file=DOSSIER_CONFIG_ET_CONSIGNES + "icone_enregistrer.png")
    bouton_enregistrer.config(image=img1)
    # img2 = PhotoImage(file= DOSSIER_CONFIG_ET_CONSIGNES + "icone_engrenage.png")
    # bouton_parametrage_consigne.config(image=img2)
    # img3 = PhotoImage(file= DOSSIER_CONFIG_ET_CONSIGNES + "icone_retour.png")
    # mise_a_0_btn.config(image=img3)
    img6 = PhotoImage(file=DOSSIER_CONFIG_ET_CONSIGNES + "icone_play.png")
    bouton_de_demarrage_du_test.config(image=img6)
    img7 = PhotoImage(file=DOSSIER_CONFIG_ET_CONSIGNES + "loss.png")
    bouton_d_arret_de_crappy.config(image=img7)
    # img8 = PhotoImage(file= DOSSIER_CONFIG_ET_CONSIGNES + "icone_tension.png")

    bouton_de_demarrage_du_test.grid(row=0, column=13, padx=5, pady=5)
    bouton_de_lancement_de_l_enregistrement.grid(row=1, column=13, padx=5, pady=5)
    bouton_de_mise_en_tension_rapide.grid(row=2, column=13, padx=5, pady=5)
    # bouton_de_mise_en_tension_lente.grid(row=3,column=13,padx =5, pady =5)
    # bouton_de_retour_en_position_initiale.grid(row=4,column=13,padx =5, pady =5)
    bouton_d_arret_de_crappy.grid(row=0, column=14, columnspan=2, padx=5, pady=5)
    bouton_enregistrer_et_quitter.grid(row=2, column=14, padx=5, pady=5)

    bouton_enregistrer.grid(row=1, column=14, padx=5, pady=5)

    fenetre_principale.mainloop()


if __name__ == "__main__":
    fonction_principale()
