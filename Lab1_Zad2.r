library(GA)

plecakDb = data.frame(
  przedmiot = c("zegar", "obraz-pejzaż","obraz-portret", "radio", "laptop", "lampka nocna", 
                "srebrne sztućce", "porcelana", "figura z brązu", "skórzana torebka", "odkurzacz"),
  wartosc = c(100, 400, 200, 40, 500, 70, 100, 250, 300, 280, 300),
  waga = c(7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15)
)
plecakLimit = 25

fitness_func = function(chr)
{
  calkowitaWartoscChr = chr %*% plecakDb$wartosc
  calkowitaWagaChr = chr %*% plecakDb$waga
  if (calkowitaWagaChr > plecakLimit)
    return(-calkowitaWartoscChr)
  else return(calkowitaWartoscChr)
}

wynik = ga(type = "binary", nBits = 11, fitness = fitness_func,
           popSize = 100, pcrossover = 0.85, pmutation = 0.05,
           elitism = 5, maxiter = 30, seed = 10, keepBest = TRUE)

summary(wynik)
plot(wynik)

decode = function(chr){
  print("Rozwiazanie: ")
  print( plecakDb[chr == 1, ] )
  print(paste("Waga plecaka =", chr %*% plecakDb$waga))
  print(paste("Wartosc przedmiotow =", chr %*% plecakDb$wartosc))
}

decode(wynik@solution[1,])
