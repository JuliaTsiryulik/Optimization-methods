from data import Point_Generator
from line import Line
from ransac import RANSAC, square_error_loss, mean_square_error

if __name__ == "__main__":
    
    point_gen = Point_Generator(100, 0.3)
    X, y = point_gen.generate_case(None, None, eps = 0.1)

    # RANSAC для 2-х точек
    ransac = RANSAC(n=2, model=Line(), loss=square_error_loss, metric=mean_square_error)
    ransac.fit(X, y)
    ransac.draw(X, y)

    # RANSAC для 5-ти точек
    ransac = RANSAC(n=5, model=Line(), loss=square_error_loss, metric=mean_square_error)
    ransac.fit(X, y)
    ransac.draw(X, y)

    # RANSAC для 15-ти точек
    ransac = RANSAC(n=15, model=Line(), loss=square_error_loss, metric=mean_square_error)
    ransac.fit(X, y)
    ransac.draw(X, y)



