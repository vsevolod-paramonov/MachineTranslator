# MachineTranslator
<hline>

> * ### [Структура репозитория](#struct)
> * ### [Checkpoint](#checkpoint)
> * ### [Итоги](#results)

<h2 name='struct'> Структура репозитория </h2>

```bash
├── README.md
├── configs ### Директория с конфигами для запуска
│   ├── checkpoint.yaml
│   ├── default.yaml
│   └── transformer.yaml
├── experiments ### Директория с результатами экспериментов
│   └── checkpoint
│       ├── de_tokenizer.model ### Сохраненные модели для токенайзера
│       ├── de_tokenizer.vocab
│       ├── en_tokenizer.model
│       ├── en_tokenizer.vocab
│       ├── logs ### Директория с сохраненными логами
│       │   └── 2025-02-28.log
│       └── test_pred.en ### Проскоренный test1.en файл
├── logger ### Директория с логгером для вывода в консоль и сохранения в .log файл
│   └── logwriter.py
├── main.py ### Файл для запуска процесса обучения
├── metrics ### Директория с метриками 
│   └── bleu.py
├── models ### Директория с моделями 
│   ├── generation.py ### Файл с методами генерации (BeamSearch + GreedyDecode)
│   ├── masks.py ### Файл с масками (PaddingMask + LookAheadMask)
│   ├── seq2seq ### Директория с реализацией RNN + Attention
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   └── translator.py
│   └── transformer ### Директория с реализацией трансформера 
│       ├── __init__.py
│       ├── decoder.py
│       ├── encoder.py
│       └── translation_transformer.py
├── mtdatasets ### Директория с файлами для обработки датасетов
│   ├── dataloader.py
│   └── dataset.py
├── trainer ### Директория с классами для обучения модели 
│   ├── base_trainer.py ### Базовый класс для процесса обучения 
│   └── translator_trainer.py ### Основной класс для обучения + валидации модели 
├── translation_dataset ### Датасет с данными
│   ├── test1.de-en.de
│   ├── train.de-en.de
│   ├── train.de-en.en
│   ├── val.de-en.de
│   └── val.de-en.en
└── utils ### Прочее 
    └── data_utils.py
```

<h2 name='checkpoint'>Checkpoint</h2>

Чтобы запустить пайплайн для обучения модели, результат которой был засабмичен для чекпоинта, необходимо выполнить следующую команду:

```bash
!python /MachineTranslator/main.py --config-name checkpoint
```
При запуске будет с нуля запущен процесс обучения и перевод для текста <code>test1.de-en.de</code>. Если же интересует только инференс с итогового 
чекпоинта после обучения всей модели, то скачать веса по [ссылке](https://disk.yandex.ru/d/SVKlu13hjUE_Og) и выполнить следующую команду, которая запустит алгоритм <code>BeamSearch</code> для перевода:

```bash
!python /MachineTranslator/main.py --config-name checkpoint +inference.test_path=/MachineTranslator/experiments/checkpoint/checkpoint_checkpoint.pth +inference.inference_mode=True
```

Файл, отправленный боту для чекпоинта, находится по пути <code>experiments/checkpoint/test_pred.en</code>. 

<h2 name='results'>Итоги</h2>

|    Config   |  #Epoch  | Train CELoss | Val CELoss | Time Fit | Time Inference | BLEU test | 
|-------------|----------|--------------|------------|----------|----------------|-----------|
|  checkpoint |    10    |      3.75    |    3.82    |  ~50min  |     ~50min     |   20.66   |
|      v2     |    10    |      3.17    |    3.44    |   ~2h    |      ~1h       |   25.78   |
|      v3     |    25    |      2.65    |    3.27    |   ~7h    |      ~1h       |   29.1    |
|     v3.2    |    30    |      2.48    |    3.23    |   ~9h    |     ~1.5h      |   29.88   |

