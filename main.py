from stock_neural_net import build_neural_net
if __name__ == '__main__':
    stocks = ['msft', 'goog']
    for name in stocks:
        build_neural_net(name)