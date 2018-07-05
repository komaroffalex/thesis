В этом проекте представлен шаблон выпускной квалификационной работы, который подойдёт как для бакалаврской, так и для магистерской.

## Работа с шаблоном

Для получения из данного шаблона **pdf** файла необходимо его собрать.

**Самый простой способ (!)** собрать данный шаблон - интегрированный в gitlab CI.
Чтобы им воспользоваться, достаточно форкнуть проект на gitlab'е.
После чего все вносимые изменения будут инициировать процесс сборки. Результатом сборки является артефакт - архив с готовым pdf.

Если же есть необходимость собирать шаблон локально, то необходимо установить какой-либо дистрибутив LaTeX.

### Установка texlive

**Ubuntu 16.04** и выше воспользуйтесь следующей командой:

```
sudo apt install texlive-latex-extra texlive-lang-cyrillic
```

**Archlinux** воспользуйтесь следующей командой:

```
sudo pacman -S texlive-bin texlive-core texlive-fontsextra texlive-formatsextra texlive-langcyrillic texlive-latexextra texlive-pictures texlive-science 
```

### Сборка шаблона

Для упрощения процесса сборки написан Makefile, использовать который можно с помощью команды:

```
make pdf
```

Результатом выполнения является файл **thesis.pdf**

Для удаления всех артефактов сборки можно выполнить команду:
```
make clean
```

## Заполнение шаблона

Для начала необходимо титульный лист, отредактировав файл `title.tex`

_____

Основано на [шаблоне](https://bitbucket.org/ice_phoenix/csse-fcs-latex/) магистерской диссертации.
