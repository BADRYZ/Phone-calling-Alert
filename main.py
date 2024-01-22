"""
Project 8: Phone calling Alert
Groupe n°1 sujet n°8;

BADRY Zakaria
MEZIANE ZERHOUNI Hassan

NB : La plupart des lignes de code ont probablement été écrites par notre équipe.
La démarche a été donnée par le moteur d'intelligence artificielle ChatGPT (partie de détection de visage et iris).
Cependant, pour la partie de détection du mouvement, nous avons regardé une vidéo qui explique une méthode et nous avons
utilisé certaines lignes de code pour calculer les distances, le cosinus et le sinus.
ce n'est pas un copier-coller du contenu.


NB2: ce code doit runner dans python 9 , avec le model dlib en bas

NB3 :a noter que les seuil suivant sont valable si la distance entre la cam et l'utilisateur peut pres de 69 cm

Références :
dlib : https://github.com/sachadee/Dlib
mask :https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
code :https://github.com/L42Project/Tutoriels/blob/master/Divers/tutoriel20/orientation.py
explication :https://www.youtube.com/watch?v=ibuEFfpVWlU

"""

#LES BIBLIOTHEQUES UTILISEES DANS L'APPLICATION
import cv2 #OPENCV POUR VIDEO / FRAME
import dlib # QUI CONTIENT LES FONCTIONS ET LE MODEL DE DETECTION DU VISAGE
import numpy as np # UTILISER POUR UTILISATION DES TABLEAUX
import time # POUR CALCULER LE TEMPS (chrono)
import math #POUR LES METHODES MATHEMATIQUES
import winsound #POUR ALERT BIP SONOR

#DECLARATION ET INISIALISATION DES VARIABLES
#VARIABLE UTILISEES POUR LA DIRECTION D'IRIS
temps_dern_pos = None #LE TEMPS DU DERNIER POSITION
pos_actuel = None # LA POSITION ACTUELLE
temps_passe= 2 #SEUIL DU TEMPS
direction=None #LA DIRECTION {DROITE/GAUCHE/CENTRE}
#VARIABLE UTILISEES POUR DETECTION DE MOUVEMENT HAUT/BAS
temps_haut= 0 #LE TEMPS PASSE EN HAUT
temps_bas = 0 #LE TEMPS PASSE EN BAS

#DEBUT DE PROGRAMME
cap = cv2.VideoCapture(0) # appel camera /0 reference a la camera principale /
detecteur = dlib.get_frontal_face_detector()# apelle la fonction qui peut detcter le visage dans un frame , predefinie dans dlib
points = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")#importer le mask de détection des points caractéristiques du visage

""" le lien du telechargement de fichier (.dat) : https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat"""

#boucle pour le traitement du fram(visage+iris+mouvement)
while True:

    ret, frame = cap.read() # Utilisez le bool ret pour vérifier si la lecture a réussi /capter la camera
    frame = cv2.flip(frame, 1)  # Inverser horizontalement la frame pour l'effet miroir
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # transformer l'image capturer en niveaux de gris
    visages = detecteur(gris)  # analyser l'image en niveaux de gris et détecter les visages dans l'image en argument(frame capter en viv gris)

    # detection visage en temps réel
    for visage_actuel in visages:
        mask = points(gris, visage_actuel)#appliquer les 68  points sur l'image en niv de gris

    # extraire les coordonnées du rectangle de visage détecté / trace le rectangle sur un visage
        x1, y1, x2, y2 = visage_actuel.left(), visage_actuel.top(), visage_actuel.right(), visage_actuel.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (127, 0, 255), 1)



        # detection des yeux
        # detecter l'œil gauche
        # ce tableau fait seulement un desin sur le conteur de l'œil (contourer l'œil)
        #on identifier la position x et y de chaque point qui caracterise l'œil a gauche de 36 jusqu'a 41
        #on peut commenter les lignes 49 jusqu'a 57 pas de contour
        gauche_contour_for_camera= np.array([
            (mask.part(36).x, mask.part(36).y),
            (mask.part(37).x, mask.part(37).y),
            (mask.part(38).x, mask.part(38).y),
            (mask.part(39).x, mask.part(39).y),
            (mask.part(40).x, mask.part(40).y),
            (mask.part(41).x, mask.part(41).y)
        ], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [gauche_contour_for_camera], True, (0, 0, 255), 1) #desin du contour


        # ce tableau fait pour le traitment de l'œil gauche
        #on identifier la position x et y de chaque point qui caracterise l'œil a gauche de 36 jusque 41

        gauche_contour = np.array([
            (mask.part(36).x, mask.part(36).y),
            (mask.part(37).x, mask.part(37).y),
            (mask.part(38).x, mask.part(38).y),
            (mask.part(39).x, mask.part(39).y),
            (mask.part(40).x, mask.part(40).y),
            (mask.part(41).x, mask.part(41).y)
        ], np.int32)
        cv2.polylines(frame, [gauche_contour], True, (0, 0, 255), 1)#desin du contour

        #explication
        """on identifier l'air du contour le l'œil a gauche grace a la fonction boundingRect qui peut encadre le contour donne et renvoie 4 variable
        x:la position selon x de le point haut a guche du rectangle , y: la position selon y de le point haut a gauche du ectangle , w:la largeur de l'œil apartir du x, h: lahauteur de l'œil a partir du y """

        gauche_aire = cv2.boundingRect(gauche_contour)

        # definir l'aire de l'œil a gauche grace aux variables resultant de boundingRect
        """les variable 0 """
        gauche= frame[gauche_aire[1]:gauche_aire[1] + gauche_aire[3], gauche_aire[0]:gauche_aire[0] + gauche_aire[2]]

        """apres detection de l'œil maintenant on detecte iris avec un seuillage , trouver le conteur ,et la methode des moments """
        ## la pritie du detection iris

        # 1-tous en gris -assurer que la region l'œil gauche est en niveau du gris
        gauche_gris = cv2.cvtColor(gauche, cv2.COLOR_BGR2GRAY)

        # 2-seuillage
        """on applique un seuillage pour identifier la partie blanche et noir l'œil- la seuil est 70"""
        _, threshold_eye = cv2.threshold(gauche_gris, 70, 255, cv2.THRESH_BINARY_INV)

        # 3-detection de contour
        #identifer tous les contours dans l'image ,pui les stocker dans une liste, les contours sont trier par un sort de hiritage
        #pere-fis / plus la constitution des contours ce fait par l'approximation
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # tri des contours ce fait avec un listement decoissante par la taille du contour, la taille de l'iris est la plus grande
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        #si on veux visualiser le conteurs de l'iris detecter
        # cv2.drawContours(gauche, [contours[0]], -1, (255, 255, 255), 2)

        # calcule le centre de l'iris
        """ce fait avec la methodes des moments qui repose sur les moments spatiaux , chaque moment est un produit de l'intensite du pixel et 
         la position de ce pixel soit en x ou en y"""
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0: #m00 est un reference de sommation de tous les pixels en x et y
                cx = int(M["m10"] / M["m00"])#m00 est un reference de sommation de tous les moment de tous les pixels en x
                cy = int(M["m01"] / M["m00"]) #m00 est un reference de sommation de tous les moment de tous les pixels en y

                #desin de centre x et y de l'iris
                cv2.circle(frame, (gauche_aire[0] + cx, gauche_aire[1] + cy), 2, (0, 0, 255), -1)

                # Calcul de la position relative de l'iris
                #par raport a le centre de l'œil
                position = cx - (gauche.shape[1] // 2)

                # Détermination de la direction du regard
                #seuil 5 pixel
                if position < -5:
                    direction = "Gauche"
                elif position > 5:
                    direction = "Droite"
                else:
                    direction = "Centre"

                cv2.putText(frame, direction, (gauche_aire [0], gauche_aire [1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
                break


        ### la droite
        """le meme traitement de l'œil gauche """
        mask = points(gris, visage_actuel)
        droite_contour_for_camera = np.array([
        (mask.part(42).x, mask.part(42).y),
        (mask.part(43).x, mask.part(43).y),
        (mask.part(44).x, mask.part(44).y),
        (mask.part(45).x, mask.part(45).y),
        (mask.part(46).x, mask.part(46).y),
        (mask.part(47).x, mask.part(47).y)], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [droite_contour_for_camera], True, (0, 0, 255), 1)

        droite_contour = np.array([
            (mask.part(42).x, mask.part(42).y),
            (mask.part(43).x, mask.part(43).y),
            (mask.part(44).x, mask.part(44).y),
            (mask.part(45).x, mask.part(45).y),
            (mask.part(46).x, mask.part(46).y),
            (mask.part(47).x, mask.part(47).y)
        ], np.int32)
        droite_aire = cv2.boundingRect(droite_contour)
        droite = frame[droite_aire[1]:droite_aire[1] + droite_aire[3],droite_aire[0]:droite_aire[0] + droite_aire[2]]
        ## detection iris
        # 1-tous en gris
        droite_gris = cv2.cvtColor(droite, cv2.COLOR_BGR2GRAY)
        # 2-seuillage,##moins 70 noire
        _, threshold_eye = cv2.threshold(droite_gris, 70, 255, cv2.THRESH_BINARY_INV)
        # detection de contour
        contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # tri
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        # calcule le centre de l'iris
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (droite_aire[0] + cx, droite_aire[1] + cy), 2, (0, 0, 255), -1)

                # Calcul de la position relative de l'iris
                centre = (droite_contour[0] + droite_contour[3]) // 2
                position = cx - (droite.shape[1] // 2)

                # Détermination de la direction du regard
                if position < -5:
                    direction = "Gauche"
                elif position > 5:
                    direction = "Droite"
                else:
                    direction = "Centre"

                cv2.putText(frame, direction, (droite_aire[0], droite_aire[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0), 1)
                break

    ### alert relative a la position gauche et droite de l'iris
    #stocker la direction actuel dans la nouvelle direction
    new_direction = direction
    if new_direction != pos_actuel:# si la nouvelle direction differente a la precedente faire :
        #remplacer direction par la nouvelle
        pos_actuel = new_direction
        #declencher le timing
        temps_dern_pos = time.time()  # Mise à jour du temps lors du changement de direction

    # Vérifier si la direction est maintenue pendant plus de 2 secondes
    if pos_actuel in ["Gauche", "Droite"] and temps_dern_pos:
        if (time.time() - temps_dern_pos) > temps_passe:
            #donner une alert sonor
            winsound.Beep(440, 500)
            # Afficher un grand "ALERT!" sur le frame
            cv2.putText(frame, "ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            print("Alerte: Direction du regard maintenue pendant plus de 2 secondes !")


    # le mouvement en haut et en bas

    """l'idee de cette partie est elaborer deja dans une video , avec un code github ,mais ous avons
    utilisé certaines lignes de code pour calculer les distances, le cosinus et le sinus. 
    
    ce n'est pas un copier-coller du contenu.
    
    https://www.youtube.com/watch?v=ibuEFfpVWlU
    https://github.com/L42Project/Tutoriels/blob/master/Divers/tutoriel20/orientation.py
    
    
    """

    """definir les variable / distance qui constituent un trangle entre les yeux et le nez """
    #distance entre les extrimiter des deux yeux
    d_eyes = math.sqrt(math.pow(mask.part(36).x - mask.part(45).x, 2) +
                       math.pow(mask.part(36).y - mask.part(45).y, 2))

    #distance entre l'extrimiter de l'oeil droite et le centre de nez
    d1 = math.sqrt(math.pow(mask.part(36).x - mask.part(30).x, 2) +
                   math.pow(mask.part(36).y - mask.part(30).y, 2))

    # distance entre l'extrimiter de l'oeil gauche et le centre de nez
    d2 = math.sqrt(math.pow(mask.part(45).x - mask.part(30).x, 2) +
                   math.pow(mask.part(45).y - mask.part(30).y, 2))

    #la somme des distance entre le nez et les deux ectrimiter
    coeff = d1 + d2
    
    """pour identifier la distance entre le centre du nez et la d-eyes il faut calculer le cos de l'angle entre ces duex lines 
    puis faire un refairence de la position normale du visage grace a a3"""

    #250 pour agrandir l'echelle

    cosb = min((math.pow(d2, 2) - math.pow(d1, 2) + math.pow(d_eyes, 2)) / (2 * d2 * d_eyes), 1)
    a3 = int(250 * (d2 * math.sin(math.acos(cosb)) - coeff / 4) / coeff)

    for n in range(0, 68):
        x = mask.part(n).x
        y = mask.part(n).y
        #tracer les points de visages les 68 points
        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        if n == 30 or n == 36 or n == 45:
            # cv2.circle(i, (x, y), 3, (255, 255, 0), -1)
            cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

        # Tracer les lignes entre les points 30 et 36
        cv2.line(frame, (mask.part(30).x, mask.part(30).y),
                 (mask.part(36).x, mask.part(36).y), (0, 255, 255), 2)

        # Tracer les lignes entre les points 45 et 30
        cv2.line(frame, (mask.part(45).x, mask.part(45).y),
                 (mask.part(30).x, mask.part(30).y), (0, 255, 255), 2)

        # Tracer les lignes entre les points 36 et 45
        cv2.line(frame, (mask.part(36).x, mask.part(36).y),
                 (mask.part(45).x, mask.part(45).y), (0, 255, 255), 2)


        ## alert relative en haut et en bas
        """a noter que les seuil suivant sont valable si la distance entre la cam et l'utilisateur peut pres de 69cm """

        txt = "la direction du regarde est :"

        #si la position a3 est inf a -1 == vue en haut
        if a3 < -1:
            txt += "en haut" #texte d'alert sur le frame
            if time.time() - temps_haut > 2:  # le regard en depasse 2s
                print("Alert: regard en haut plus de 2 seconds!")#texte d'alert sur la console
                winsound.Beep(440, 500)#alert sonore
                cv2.putText(frame, "ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

            #renitialiser timing
            temps_haut = time.time()


        # le meme logique pour regarde en bas
        elif a3 > 20:
            txt += "en bas"
            if time.time() - temps_bas > 2:
                print("Alert: regard en bas plus de 2 seconds!")
                winsound.Beep(440, 500)
                cv2.putText(frame, "ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)



            temps_bas = time.time()  # Update the timer
        else:
            txt += "la camera "


    #afficher les texte de la position et d'alert sur le frame
    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 255), 2)

    #la fenetre du frame
    cv2.imshow("Frame", frame)

    #si on clique q on quite le programme

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
