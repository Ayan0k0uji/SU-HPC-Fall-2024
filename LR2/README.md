Задача разделяется на подзадачи. Каждая подзадача выполняется в своем блоке на GPU. В блоке каждая нить получает свой элемент данных для суммирования.

В блоках используется стратегия разделяй и влавствуй (суммирование данных по шагам). На каждом шаге нити суммируют данные с соседями до тех пор, пока не останется одна сумма(Сумма всего блока, которая записывается в глобальную память в выходной массив).

Весь массив делится на блоки с заданным количеством элементов(256). Блоки обрабатываются независимо.

Потоки в блоке загружают по одному элементу данных в общую память. Далее используется стратегия разделяй и влавствуй (суммирование) с логарифмическим количеством шагов. Это позволяет суммировать элементы в блоке за меньшее количество шагов.

| N                   | Time CPU (sec)  | Time GPU (sec)  | Boost               |
|---------------------|-----------------|-----------------|---------------------|
| 1000                | 0.000006        | 0.000109        | 0,055               |
| 10000               | 0.000043        | 0.000138        | 0.312               |
| 100000              | 0.000642        | 0.000313        | 2.051               |
| 1000000             | 0.004538        | 0.001968        | 2.305               |

