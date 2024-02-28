import stanza # importar stanza 

nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma') # Se crea un objeto Pipeline de Stanza para el procesamiento de texto en español. Se especifican los procesadores que se utilizarán en el pipeline, que incluyen tokenización, multi-word token expansion (mwt), etiquetado de partes del discurso (pos) y lematización (lemma).

with open('pinocho.txt', 'r', encoding='utf-8') as file: # Abre el archivo de texto llamado 'pinocho.txt' en modo lectura ('r') y lo asocia a la variable file. Se especifica el argumento encoding='utf-8' para asegurarse de que el archivo se lea correctamente en UTF-8, lo que es importante para manejar caracteres especiales en español.
    text = file.read() # Lee todo el contenido del archivo 'pinocho.txt' y lo almacena en la variable text.

doc = nlp(text) #Procesa el texto utilizando el pipeline creado anteriormente (nlp) y lo almacena en la variable doc. Esto ejecutará la tokenización, expansión de tokens multi-palabra, etiquetado de partes del discurso y lematización en el texto.

with open('pinocho_result.txt', 'w', encoding='utf-8') as file: # Abre (o crea si no existe) un archivo de texto llamado 'pinocho_result.txt' en modo escritura ('w') y lo asocia a la variable file. Se utiliza el argumento encoding='utf-8' nuevamente para asegurar la escritura adecuada de caracteres especiales.
    for i, sentence in enumerate(doc.sentences): # Itera sobre cada oración procesada en el documento (doc). La función enumerate() agrega un contador a cada oración, comenzando desde 0, y devuelve tanto el índice como la oración misma.
        file.write(f"====== Frase {i+1} tokens ======\n") #Formato de archivo.
        for j, word in enumerate(sentence.words): # Itera sobre cada palabra procesada en la oración actual. enumerate() se utiliza nuevamente para contar las palabras en cada oración.
            file.write(f"id: {j+1} Palabra: {word.text}\t\tLema: {word.lemma}\n") # Escribe en el archivo la información de cada palabra, incluyendo su posición (índice), forma original (text) y lema (lemma). Se utiliza el formato de cadena f-string para incluir variables dentro de la cadena.
        file.write('\n') # Escribe una línea en blanco después de todas las palabras de una oración.

print("Tokenización y lematización completadas. Los resultados se han guardado en 'pinocho_result.txt'.") # Imprime un mensaje en la consola indicando que el proceso de tokenización y lematización ha finalizado correctamente.
