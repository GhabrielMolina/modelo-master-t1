#!/usr/bin/env python3
"""
Simulador de rede de filas com topologia generica.

Simulacao a eventos discretos de uma rede de filas G/G/c/K, onde cada fila
pode ter caracteristicas proprias (numero de servidores, capacidade,
intervalos de chegada e de atendimento) e o roteamento entre as filas e
definido por probabilidades. Instrucoes de uso no README.md.

A saida no console e em ASCII puro de proposito: o console do Windows
costuma usar cp1252 e embaralharia acentos.
"""

import argparse
import heapq
import json
import sys

EPSILON = 1e-9
CHEGADA = "CHEGADA"
SAIDA = "SAIDA"


class SimulacaoEncerrada(Exception):
    """Esgotou a quantidade de aleatorios contratada.

    Interromper o tratamento de um evento no meio nao corrompe as
    estatisticas, porque o tempo e contabilizado ANTES do tratamento.
    """


class ErroDeConfiguracao(Exception):
    pass


class Gerador:
    """Gerador congruente linear: X(n+1) = (a * X(n) + c) mod M.

    Entrega no maximo `quantidade_maxima` valores; o pedido seguinte encerra
    a simulacao. Esse e o criterio de parada do trabalho: "ao se utilizar o
    100.000 aleatorio, a simulacao deve se encerrar".
    """

    def __init__(self, semente, a, c, M, quantidade_maxima):
        self.a = a
        self.c = c
        self.M = M
        self.quantidade_maxima = quantidade_maxima
        self.estado = semente
        self.usados = 0
        # Aleatorios sorteados durante o evento corrente, apenas para o trace.
        self.registro = []

    def proximo(self):
        if self.usados >= self.quantidade_maxima:
            raise SimulacaoEncerrada()
        self.estado = (self.a * self.estado + self.c) % self.M
        self.usados += 1
        valor = self.estado / self.M
        self.registro.append(valor)
        return valor

    def uniforme(self, minimo, maximo):
        return minimo + (maximo - minimo) * self.proximo()


class Fila:
    """Um no da rede: uma fila G/G/servidores/capacidade.

    `status` e o numero de clientes NO SISTEMA (em espera + em atendimento).
    `tempos` acumula, para cada valor de status, o tempo total em que a fila
    permaneceu naquele estado.
    """

    def __init__(self, nome, servidores, capacidade, chegada, atendimento):
        self.nome = nome
        self.servidores = servidores
        self.capacidade = capacidade      # None = capacidade infinita
        self.chegada = chegada            # (min, max), ou None se nao ha
        self.atendimento = atendimento    # (min, max)

        # Opcoes de roteamento: lista de (Fila ou None, probabilidade).
        # None representa "sair do sistema". Montada em carrega_config().
        self.opcoes = []

        self.status = 0
        self.perdas = 0
        self.tempos = {}

    def tem_chegadas_externas(self):
        return self.chegada is not None

    def pode_receber(self):
        return self.capacidade is None or self.status < self.capacidade

    def atende_imediatamente(self):
        """Chamado APOS incrementar o status: havia servidor livre?"""
        return self.status <= self.servidores

    def ha_cliente_esperando(self):
        """Chamado APOS decrementar o status: sobrou alguem na espera?"""
        return self.status >= self.servidores

    def contabiliza(self, delta):
        self.tempos[self.status] = self.tempos.get(self.status, 0.0) + delta

    def kendall(self):
        return "G/G/%d/%s" % (
            self.servidores,
            "inf" if self.capacidade is None else self.capacidade)

    def distribuicao(self, tempo_global):
        """(estado, tempo acumulado, probabilidade em %) para cada estado.

        Vai de 0 ate K, ou ate o maior estado observado se a capacidade e
        infinita.
        """
        if self.capacidade is not None:
            maximo = self.capacidade
        else:
            maximo = max(self.tempos) if self.tempos else 0

        linhas = []
        for estado in range(maximo + 1):
            tempo = self.tempos.get(estado, 0.0)
            probabilidade = (tempo / tempo_global * 100.0) if tempo_global > 0 else 0.0
            linhas.append((estado, tempo, probabilidade))
        return linhas


class Configuracao:
    """Parametros da simulacao lidos do arquivo JSON.

    Os escalares sao extraidos e validados aqui; `filas` e `chegadas_iniciais`
    sao preenchidos em seguida por carrega_config().
    """

    def __init__(self, caminho, dados):
        self.caminho = caminho
        self.semente = int(dados.get("semente", 1))

        self.quantidade_aleatorios = int(dados.get("quantidadeAleatorios", 100000))
        if self.quantidade_aleatorios < 1:
            raise ErroDeConfiguracao("'quantidadeAleatorios' deve ser maior que zero.")

        parametros = dados.get("gerador", {})
        self.gerador_a = int(parametros.get("a", 1664525))
        self.gerador_c = int(parametros.get("c", 1013904223))
        self.gerador_M = int(parametros.get("M", 2 ** 32))
        if self.gerador_M < 2:
            raise ErroDeConfiguracao("O modulo 'M' do gerador deve ser maior que 1.")

        self.sortear_rota_deterministica = bool(
            dados.get("sortearRotaDeterministica", False))

        self.filas = {}                 # nome -> Fila (ordem do arquivo)
        self.chegadas_iniciais = {}     # nome -> instante da primeira chegada


def _numero(valor, campo):
    if not isinstance(valor, (int, float)) or isinstance(valor, bool):
        raise ErroDeConfiguracao("O campo '%s' deve ser numerico." % campo)
    return float(valor)


def _intervalo(descricao, nome, prefixo):
    """Le o par '<prefixo>Min'/'<prefixo>Max' de uma fila, ambos presentes."""
    campo_min = prefixo + "Min"
    campo_max = prefixo + "Max"
    minimo = _numero(descricao[campo_min], "%s.%s" % (nome, campo_min))
    maximo = _numero(descricao[campo_max], "%s.%s" % (nome, campo_max))
    if minimo > maximo:
        raise ErroDeConfiguracao(
            "Na fila '%s', '%s' e maior que '%s'." % (nome, campo_min, campo_max))
    return minimo, maximo


def _le_filas(config, dados):
    descricoes = dados.get("filas")
    if not isinstance(descricoes, dict) or not descricoes:
        raise ErroDeConfiguracao(
            "A configuracao deve conter um objeto 'filas' com ao menos uma fila.")

    for nome, descricao in descricoes.items():
        if not isinstance(descricao, dict):
            raise ErroDeConfiguracao("A fila '%s' deve ser um objeto." % nome)

        servidores = int(descricao.get("servidores", 1))
        if servidores < 1:
            raise ErroDeConfiguracao(
                "A fila '%s' precisa de ao menos 1 servidor." % nome)

        capacidade = descricao.get("capacidade", None)
        if capacidade is not None:
            capacidade = int(capacidade)
            if capacidade < servidores:
                raise ErroDeConfiguracao(
                    "A fila '%s' tem capacidade %d menor que seus %d servidores."
                    % (nome, capacidade, servidores))

        if "atendimentoMin" not in descricao or "atendimentoMax" not in descricao:
            raise ErroDeConfiguracao(
                "A fila '%s' precisa de 'atendimentoMin' e 'atendimentoMax'." % nome)
        atendimento = _intervalo(descricao, nome, "atendimento")

        # Chegadas externas sao opcionais, mas o par tem que vir completo.
        if ("chegadaMin" in descricao) != ("chegadaMax" in descricao):
            raise ErroDeConfiguracao(
                "A fila '%s' deve informar 'chegadaMin' e 'chegadaMax' juntos, "
                "ou nenhum dos dois (fila sem chegadas externas)." % nome)
        chegada = None
        if "chegadaMin" in descricao:
            chegada = _intervalo(descricao, nome, "chegada")

        config.filas[nome] = Fila(nome, servidores, capacidade,
                                  chegada, atendimento)


def _le_rede(config, dados):
    rotas_por_origem = {}
    for aresta in dados.get("rede", []):
        origem = aresta.get("origem")
        destino = aresta.get("destino")
        probabilidade = _numero(aresta.get("probabilidade", 0.0), "probabilidade")

        if origem not in config.filas:
            raise ErroDeConfiguracao(
                "A rota cita a origem '%s', que nao existe em 'filas'." % origem)
        if destino not in config.filas:
            raise ErroDeConfiguracao(
                "A rota %s -> '%s' cita um destino que nao existe em 'filas'."
                % (origem, destino))
        if probabilidade < 0.0 or probabilidade > 1.0:
            raise ErroDeConfiguracao(
                "A rota %s -> %s tem probabilidade %s fora do intervalo [0, 1]."
                % (origem, destino, probabilidade))
        rotas_por_origem.setdefault(origem, []).append((destino, probabilidade))

    for nome, fila in config.filas.items():
        rotas = rotas_por_origem.get(nome, [])
        soma = sum(probabilidade for _, probabilidade in rotas)
        if soma > 1.0 + EPSILON:
            raise ErroDeConfiguracao(
                "As rotas que saem da fila '%s' somam %.6f, que excede 1.0."
                % (nome, soma))
        fila.opcoes = [(config.filas[destino], probabilidade)
                       for destino, probabilidade in rotas]
        # O que sobra ate 1.0 e a probabilidade de o cliente ir embora.
        saida_externa = 1.0 - soma
        if saida_externa > EPSILON:
            fila.opcoes.append((None, saida_externa))


def _le_chegadas_iniciais(config, dados):
    iniciais = dados.get("chegadasIniciais", {})
    if not isinstance(iniciais, dict):
        raise ErroDeConfiguracao("'chegadasIniciais' deve ser um objeto.")
    for nome, instante in iniciais.items():
        if nome not in config.filas:
            raise ErroDeConfiguracao(
                "'chegadasIniciais' cita a fila '%s', que nao existe." % nome)
        config.chegadas_iniciais[nome] = _numero(instante,
                                                 "chegadasIniciais.%s" % nome)

    # Ter chegadas externas e aparecer em 'chegadasIniciais' sao a mesma coisa.
    for nome, fila in config.filas.items():
        tem_inicial = nome in config.chegadas_iniciais
        if fila.tem_chegadas_externas() and not tem_inicial:
            raise ErroDeConfiguracao(
                "A fila '%s' tem chegadas externas mas nao aparece em "
                "'chegadasIniciais'." % nome)
        if tem_inicial and not fila.tem_chegadas_externas():
            raise ErroDeConfiguracao(
                "A fila '%s' aparece em 'chegadasIniciais' mas nao define "
                "'chegadaMin'/'chegadaMax'." % nome)

    if not config.chegadas_iniciais:
        raise ErroDeConfiguracao(
            "Nenhuma fila recebe clientes do exterior: a simulacao nunca comecaria.")


def carrega_config(caminho):
    """Le e valida o arquivo JSON de configuracao da rede."""
    try:
        with open(caminho, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
    except FileNotFoundError:
        raise ErroDeConfiguracao("Arquivo de configuracao nao encontrado: %s" % caminho)
    except json.JSONDecodeError as erro:
        raise ErroDeConfiguracao("JSON invalido em %s: %s" % (caminho, erro))

    config = Configuracao(caminho, dados)
    _le_filas(config, dados)
    _le_rede(config, dados)
    _le_chegadas_iniciais(config, dados)
    return config


class Simulacao:
    """Simulacao a eventos discretos da rede.

    Ha apenas dois tipos de evento: CHEGADA (cliente vindo do exterior) e
    SAIDA (cliente termina o atendimento em uma fila). O destino do cliente
    que sai e resolvido no momento em que o evento e PROCESSADO, o que faz
    realimentacao e ciclos funcionarem sem nenhum codigo adicional.
    """

    def __init__(self, config, gerador, limite_trace=0):
        self.config = config
        self.filas = config.filas
        self.gerador = gerador
        self.tempo = 0.0
        self.agenda = []
        self.sequencia = 0          # desempate FIFO de eventos simultaneos
        self.limite_trace = limite_trace
        self.trace = []
        self.eventos_processados = 0

    def agendar(self, instante, tipo, fila):
        heapq.heappush(self.agenda, (instante, self.sequencia, tipo, fila.nome))
        self.sequencia += 1

    def agenda_chegada(self, fila):
        if not fila.tem_chegadas_externas():
            return
        self.agendar(self.tempo + self.gerador.uniforme(*fila.chegada),
                     CHEGADA, fila)

    def agenda_atendimento(self, fila):
        self.agendar(self.tempo + self.gerador.uniforme(*fila.atendimento),
                     SAIDA, fila)

    def contabiliza_tempo(self, instante):
        """O tempo e contabilizado ANTES do tratamento de cada evento."""
        delta = instante - self.tempo
        for fila in self.filas.values():
            fila.contabiliza(delta)
        self.tempo = instante

    def entra(self, fila):
        """Admite um cliente na fila; se ela esta cheia, contabiliza a perda."""
        if not fila.pode_receber():
            fila.perdas += 1
            return
        fila.status += 1
        if fila.atende_imediatamente():
            self.agenda_atendimento(fila)

    def chega(self, fila):
        self.entra(fila)
        self.agenda_chegada(fila)

    def sai(self, origem):
        origem.status -= 1
        # A origem e reagendada ANTES de tratar o destino: essa ordem define a
        # sequencia de consumo dos aleatorios e segue o pseudocodigo classico.
        if origem.ha_cliente_esperando():
            self.agenda_atendimento(origem)

        destino = self.sorteia_destino(origem)
        if destino is not None:         # None = cliente vai embora do sistema
            self.entra(destino)         # perda contabilizada no DESTINO

    def sorteia_destino(self, origem):
        """Escolhe para onde o cliente vai. None = sair do sistema.

        Quando ha uma unica opcao (probabilidade 1), nenhum aleatorio e
        consumido: sortear seria desperdicar um terco dos aleatorios em
        decisoes cujo resultado ja e conhecido. O flag de configuracao
        'sortearRotaDeterministica' inverte esse comportamento.
        """
        opcoes = origem.opcoes
        if not opcoes:
            return None
        if len(opcoes) == 1 and not self.config.sortear_rota_deterministica:
            return opcoes[0][0]

        sorteio = self.gerador.proximo()
        acumulado = 0.0
        for destino, probabilidade in opcoes:
            acumulado += probabilidade
            if sorteio < acumulado:
                return destino
        return opcoes[-1][0]            # protege contra erro de arredondamento

    def executar(self):
        for nome, instante in self.config.chegadas_iniciais.items():
            # A primeira chegada e DADA pelo enunciado, nao sorteada:
            # ela nao consome nenhum aleatorio.
            self.agendar(instante, CHEGADA, self.filas[nome])

        try:
            while self.agenda:
                instante, _, tipo, nome = heapq.heappop(self.agenda)
                fila = self.filas[nome]
                self.contabiliza_tempo(instante)
                self.gerador.registro = []

                if tipo == CHEGADA:
                    self.chega(fila)
                else:
                    self.sai(fila)

                self.eventos_processados += 1
                if len(self.trace) < self.limite_trace:
                    estados = {n: f.status for n, f in self.filas.items()}
                    self.trace.append((instante, tipo, nome,
                                       self.gerador.registro, estados))
        except SimulacaoEncerrada:
            pass


def _ok(condicao):
    return "OK" if condicao else "!!"


def formata_relatorio(simulacao):
    config = simulacao.config
    tempo_global = simulacao.tempo
    usados = simulacao.gerador.usados
    linhas = []
    add = linhas.append

    add("=" * 61)
    add("  SIMULADOR DE REDE DE FILAS")
    add("  Config: %s | Semente: %d" % (config.caminho, config.semente))
    add("=" * 61)

    for fila in config.filas.values():
        add("")
        add("*" * 61)
        add("Fila: %s (%s)" % (fila.nome, fila.kendall()))
        if fila.tem_chegadas_externas():
            add("Chegadas: %g ... %g" % fila.chegada)
        else:
            add("Chegadas: somente da rede (sem chegadas externas)")
        add("Atendimento: %g ... %g" % fila.atendimento)
        add("Roteamento: %s" % ", ".join(
            "%s %.2f%%" % (destino.nome if destino else "SAIDA_DO_SISTEMA",
                           probabilidade * 100.0)
            for destino, probabilidade in fila.opcoes))
        add("*" * 61)
        add("   Estado        Tempo         Probabilidade")
        add("   " + "-" * 42)
        for estado, tempo, probabilidade in fila.distribuicao(tempo_global):
            add("   %6d   %16.4f   %11.2f%%" % (estado, tempo, probabilidade))
        add("   " + "-" * 42)
        add("   Perda de clientes: %d" % fila.perdas)

    add("")
    add("=" * 61)
    add("Tempo global da simulacao: %.4f" % tempo_global)
    add("Aleatorios utilizados: %d" % usados)
    add("Eventos processados: %d" % simulacao.eventos_processados)
    add("=" * 61)

    add("")
    add("Verificacoes de sanidade")
    add("-" * 61)
    esperado = config.quantidade_aleatorios
    add("  [%s] aleatorios utilizados = %d (esperado %d)"
        % (_ok(usados == esperado), usados, esperado))
    for fila in config.filas.values():
        distribuicao = fila.distribuicao(tempo_global)
        soma_prob = sum(probabilidade for _, _, probabilidade in distribuicao)
        soma_tempo = sum(tempo for _, tempo, _ in distribuicao)
        add("  [%s] %s: probabilidades somam %.4f%% e tempos somam %.4f"
            % (_ok(abs(soma_prob - 100.0) < 1e-6
                   and abs(soma_tempo - tempo_global) < 1e-6),
               fila.nome, soma_prob, soma_tempo))
    add("-" * 61)

    return "\n".join(linhas)


def formata_trace(trace):
    linhas = ["", "Trace dos %d primeiros eventos" % len(trace), "-" * 61]
    for indice, evento in enumerate(trace, start=1):
        instante, tipo, fila, aleatorios, estados = evento
        resumo = " ".join("%s=%d" % (nome, status)
                          for nome, status in estados.items())
        sorteados = (", ".join("%.6f" % valor for valor in aleatorios)
                     or "nenhum")
        linhas.append("%4d) t=%10.4f  %-8s %-8s | estados: %s"
                      % (indice, instante, tipo, fila, resumo))
        linhas.append("      aleatorios sorteados: %s" % sorteados)
    linhas.append("-" * 61)
    return "\n".join(linhas)


def principal(argumentos=None):
    analisador = argparse.ArgumentParser(
        description="Simulador de rede de filas com topologia generica.")
    analisador.add_argument(
        "config", help="caminho do arquivo JSON que descreve a rede de filas")
    analisador.add_argument(
        "--semente", type=int, default=None,
        help="sobrepoe a semente definida no arquivo de configuracao")
    analisador.add_argument(
        "--aleatorios", type=int, default=None,
        help="sobrepoe a quantidade de numeros aleatorios da simulacao")
    analisador.add_argument(
        "--trace", type=int, default=0, metavar="N",
        help="imprime os N primeiros eventos, para conferencia manual")
    opcoes = analisador.parse_args(argumentos)

    try:
        config = carrega_config(opcoes.config)
    except ErroDeConfiguracao as erro:
        print("ERRO DE CONFIGURACAO: %s" % erro)
        return 1

    if opcoes.semente is not None:
        config.semente = opcoes.semente
    if opcoes.aleatorios is not None:
        if opcoes.aleatorios < 1:
            print("ERRO: --aleatorios deve ser maior que zero.")
            return 1
        config.quantidade_aleatorios = opcoes.aleatorios

    gerador = Gerador(config.semente, config.gerador_a, config.gerador_c,
                      config.gerador_M, config.quantidade_aleatorios)
    simulacao = Simulacao(config, gerador, opcoes.trace)
    simulacao.executar()

    if simulacao.trace:
        print(formata_trace(simulacao.trace))
    print(formata_relatorio(simulacao))

    return 0


if __name__ == "__main__":
    sys.exit(principal())
