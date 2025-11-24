def calcularNivel(pontos, maximo):
    if pontos >= maximo:
        print("Você já é um mestre de D&D!")
    else:
        print("Continue jogando para subir de nível!")
    return pontos + 2
def main():
    nome = input("Qual seu codinome no Hell Fire Club? ")
    pontos = 0
    maximo = 12
    while pontos < maximo:
        print(f"{nome}, você tem {pontos} pontos.")
        pontos = calcularNivel(pontos, maximo)
    print(f"Parabéns, {nome}! Você chegou ao nível {pontos}.")
if __name__ == "__main__":
    main()
