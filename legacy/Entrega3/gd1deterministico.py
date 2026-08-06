

class gd1:

    def calcular(self, pol, itera, umbral, a, b, alfa,X0):
        resul={}
        termina = False
        x=X0
        numIter=0
        while(not termina):
            numIter = numIter+1

            y=pol.evalPol(x)


            #Actualizar
            print "x: "+str(x)
            xn = x - alfa * pol.evalPolD(x)
            print "xn: "+str(xn)
            epsilon = abs(xn-x)
            resul["Iteracion "+str(numIter)] = {'x':x,'f(x)':y,'error':epsilon}
            print resul
            x=xn

            if numIter > itera or epsilon <= umbral:
                termin=True

        return resul
