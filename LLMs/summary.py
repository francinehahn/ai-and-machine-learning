from transformers import pipeline

summary = pipeline("summarization") #the standard model will be installed
text = """Carl Edward Sagan (Nova Iorque, 9 de novembro de 1934 — Seattle, 20 de dezembro de 1996) foi um cientista planetário, 
astrônomo, astrobiólogo, astrofísico, escritor, divulgador científico e ativista norte-americano. Sagan é autor de mais de 600 
publicações científicas e também de mais de vinte livros de ciência e ficção científica. Foi durante a vida um grande defensor 
do ceticismo e do uso do método científico. Promoveu a busca por inteligência extraterrestre através do projeto SETI e instituiu 
o envio de mensagens a bordo de sondas espaciais, destinadas a informar possíveis civilizações extraterrestres sobre a existência 
humana. Mediante suas observações da atmosfera de Vênus, foi um dos primeiros cientistas a estudar o efeito estufa em escala 
planetária. Também fundou a organização não governamental Sociedade Planetária e foi pioneiro no ramo da exobiologia. Sagan passou 
grande parte da carreira como professor da Universidade Cornell, onde foi diretor do laboratório de estudos planetários. 
Em 1960 obteve o título de doutor pela Universidade de Chicago"""

result = summary(text, max_length=100, min_length=50)
print(result)