library(GA)

fitness_func <- function(x1, x2) {
  return((2*x1+x2^2)-4)
}

min_x1 = -4
min_x2 = -12
max_x1 = 2
max_x2 = 8.5

x1 <- seq(min_x1, max_x1, by=0.8)
x2 <- seq(min_x2, max_x2, by=0.5)
z <-outer(x1, x2, fitness_func)
persp(x1, x2, z, theta=30, phi=30, expand=0.5, col=7)
image(x1, x2, z, col=terrain.colors(50))
contour(x1, x2, z, add=T, col="darkblue", nlevels=50)

monitor_func <- function(obj) {
  contour(x1, x2, z, col="magenta", nlevels=50,
          main=paste("iteration = ", obj@iter))
  points(obj@population, pch=20)
}

wyniki<-ga(type="real-valued",lower=c(min_x1, min_x2),upper=c(max_x1, max_x2),
           fitness=function(x) fitness_func(x[1],x[2]),monitor=monitor_func,
           popSize=20,pcrossover=0.85,pmutation=0.05,
           elitism=5,maxiter=120,seed=10, keepBest = TRUE)
summary(wyniki)
plot(wyniki)
