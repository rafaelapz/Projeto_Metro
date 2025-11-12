#linhas atuais / em operação
azul1 = [114,67,116,115,117,75,11,37,59,88,96,58,89,103,68,7,109,82,76,95,90,32,51]
verde2 = [108, 87,31, 70, 101,20,68, 7, 29, 86, 6, 79, 98,111]
vermelha3 = [66,60,80, 78, 9,96,71, 17, 19,14, 99, 28, 72, 110, 47,69, 12, 35]
amarela4 = [59, 78, 48, 70, 65, 43, 41, 74, 23,93,112]
lilas5 = [27,26,106,45,85,57,3,5,16,21,25,39,61,2,49,82,29]
rubi7 = [118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,66,59, 17]
diamante8 = [157,158,159,160,161,162,163,164,165,166,167,168,169,170, 171,172,173,174,175,132, 66, 176]
esmeralda9 = [172, 173, 177, 178, 179, 74, 181, 182, 183, 184, 185, 186, 85,187, 188, 189, 190, 191,192, 193]
turquesa10 = [59, 17, 146, 147, 98, 148, 149, 150, 151, 152, 153, 154, 155, 156]
coral11 = [59,17,99,35,135,136,137,138,139,140,204,141,142,143,144,145]
safira12 = [17, 99, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204]
jade13 = [194,205, 206]
prata15 = [111, 63, 91,24, 113,38, 55, 94, 42,92, 54]

# expressos
coral11_expresso_leste = [59, 137, 145]
expresso_aeroporto = [194,205, 206]

# linhas em construção
laranja6 = [18,104, 50, 56,44, 84,133, 97, 73,77, 40, 48,1,13,89]
ouro17 = [184, 30, 105, 25, 102, 22]
ouro17_ramal_congonhas = [22, 33]
ouro17_ramal_washington_luis = [22, 53]

# expansoes em construção (ou com construção confirmada)
expansao_verde2 = [64, 81, 8,107, 83, 46, 10, 72]
expansao_prata15_sentido_oratorio = [147,111]
expansao_prata15_sentido_jardim_colonial = [54, 15, 52, 207, 208, 209, 210]
expansao_amarela4 = [112, 36, 100]
expansao_lilas5 = [27,211, 212]


#alta chance de ser contruida / licitacao ativa ou definida
# localizacao das estacoes do decreto de desapropriacao
#disponivel em https://transparencia.metrosp.com.br/sites/default/files/DECRETO%20Nº%2068.537%2C%20DE%2020%20DE%20MAIO%20DE%202024.pdf
# https://transparencia.metrosp.com.br/dataset/decreto-nº-68537-de-utilidade-pública-da-linha-19-celeste-trecho-entre-praça-pedro-lessa-e
celeste19_fase1 = []
celeste19_fase2 = []
celeste19_fase3 = []

'''
Bosque Maia – Av. Tiradentes, 1451 (Guarulhos)
Guarulhos Centro – Rua Dom Pedro II, 148/154
Vila Augusta – Av. Guarulhos, 604
Dutra – Rua Segundo Tenente Aviador John Richardson Cordeiro e Silva, 220 (Internacional Shopping Guarulhos) - EXISTE, LINHA 2
Itapegica – Rua Cavadas, 247, 1624
Jardim Julieta – Av do Poeta, 929
Vila Sabrina – Av. João Simão de Castro, 250
Cerejeiras – Av. das Cerejeiras, 1950
Santo Eduardo – Praça Santo Eduardo, 57
Vila Maria – Avenida Guilherme Cotching, 563
Catumbi – Rua Marcos Arruda, 909
Silva Teles – Rua Santa Rita, 70
Cerealista – Rua Mendes Caldeira, 223 - EXISTE, LINHA 10 E 11
São Bento – Rua do Seminário, 87 - EXISTE, LINHA 1
Anhangabaú – Rua Santo Amaro, 43/45/47/53/61/67 -EXISTE, LINHA 3
fonte: https://transparencia.metrosp.com.br/sites/default/files/DECRETO%20Nº%2068.537%2C%20DE%2020%20DE%20MAIO%20DE%202024.pdf
'''


#prioritizada pelo governo / licitacao anunciada para esse ano
violeta16 =[]

#MÉDIA PRIORIDADE - EM ELABORAÇÃO DE PROJETO
rosa20 = []
marrrom22 = []

#expansao planejada (em estudo) 
expansao_esmeralda9 = []
expansao_ouro17 = [] 

#projeto  em estudo inicial
limao23 = []
onix14 = []
quartzo24 = []
topazio25 = []

#deixou de constar nos projetos
grafite21 = []

#projeto desistido 
bronze18 =[]

# Group the lines
current_and_express_lines = [
    azul1, verde2, vermelha3, amarela4, lilas5, diamante8, 
    esmeralda9, turquesa10, coral11, safira12, jade13, prata15, rubi7,
    coral11_expresso_leste, expresso_aeroporto
]

current_express_and_under_construction_lines = current_and_express_lines + [laranja6 ,ouro17 ,ouro17_ramal_congonhas,ouro17_ramal_washington_luis,
                                                                            expansao_verde2 ,expansao_prata15_sentido_oratorio,expansao_prata15_sentido_jardim_colonial,
                                                                            expansao_amarela4, expansao_lilas5 
]

all_lines = current_and_express_lines + [
    laranja6, ouro17, ouro17_ramal_congonhas, ouro17_ramal_washington_luis,
    expansao_verde2, expansao_prata15_sentido_oratorio, 
    expansao_prata15_sentido_jardim_colonial, expansao_amarela4, expansao_lilas5
]