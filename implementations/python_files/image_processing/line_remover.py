"""
    Authors: Lucas Correia e Wilson Neto
    This module is dedicated to the procedures implemented by Lucas and Wilson to remove lines
    in an image of an essay
"""
import cv2
import numpy as np


# As funções calculam a intensidades média para cada linha ou coluna da imagem
# e retornam as possíveis linhas horizontais ou verticais baseada no limiar informado.
# Para obter-se o resultado esperado pode ser necessário ordenar todas as linhas
# obtidas baseado em alguma característica, como espessura ou posição.

def verticalLines(image, thresh, totalLines=None, sort='thickness', reverse=True):
    vals = np.array([s for s in [np.sum(image[:, y]) / (255 * image.shape[0]) for y in range(image.shape[1])]])
    lines = []
    y = 0
    while y < image.shape[1]:
        if vals[y] < thresh:
            start = y
            y += 1
            while y < image.shape[1] - 1 and vals[y] < thresh:
                y += 1
            l = {}
            l['pos'] = (start, y)
            l['thickness'] = y - start
            l['mean'] = np.mean(vals[start:y])
            lines.append(l)
        y += 1
    if sort == 'thickness':
        lines.sort(key=lambda x: x[sort], reverse=reverse)
    else:
        lines.sort(key=lambda x: x[sort], reverse=reverse)
    if totalLines == None:
        return lines
    else:
        return lines[:min(totalLines, len(lines))]


def horizontalLines(image, thresh, totalLines=None, sort='thickness', reverse=True):
    vals = np.array([s for s in [np.sum(image[x, :]) / (255 * image.shape[1]) for x in range(image.shape[0])]])
    lines = []
    x = 0
    while x < image.shape[0]:
        if vals[x] < thresh:
            start = x
            x += 1
            while x < image.shape[0] - 1 and vals[x] < thresh:
                x += 1
            l = {}
            l['pos'] = (start, x)
            l['thickness'] = x - start
            l['mean'] = np.mean(vals[start:x])
            lines.append(l)
        x += 1
    if sort == 'thickness':
        lines.sort(key=lambda x: x[sort], reverse=reverse)
    else:
        lines.sort(key=lambda x: x[sort], reverse=reverse)
    if totalLines == None:
        return lines
    else:
        return lines[:min(totalLines, len(lines))]


def findLinesInEssays(essays):
    # Tentando detectar as posições das 4 linhas mais grossas
    LINES = {}
    for e in essays:
        # gray = cv2.cvtColor(essays[e],cv2.COLOR_BGR2GRAY)
        gray = essays[e].copy()
        for t in np.arange(0.4, 0.9, 0.1):
            hlines = horizontalLines(gray, t, 2)
            if len(hlines) == 2:
                break
        if len(hlines) != 2:
            print('Linhas horizontais delimitantes não encontradas (' + str(len(hlines)) + '/' + '2)')
            print('essay: ' + e)
            return {}
        thresh = [t]
        for t in np.arange(0.4, 0.9, 0.1):
            vlines = verticalLines(gray, t, 2)
            if len(vlines) == 2:
                break
        if len(vlines) != 2:
            print('Linhas verticais delimitantes não encontradas (' + str(len(vlines)) + '/' + '2)')
            print('essay: ' + e)
            return {}
        thresh.append(t)

        LINES[e] = {}
        LINES[e]['vertical'] = vlines
        LINES[e]['horizontal'] = hlines

        # Tentando detectar o restante
        evlines = [v['pos'] for v in LINES[e]['vertical']]
        evlines.sort()
        # print(evlines)
        # tenta cortar a numeração
        cropV = int(np.round((evlines[1][0] - evlines[0][1]) / 50))
        start = evlines[0][1] + cropV
        end = evlines[1][0] - cropV
        for t in np.arange(0.4, 0.9, 0.1):
            vlines = verticalLines(gray[:, start:end], t, 1)
            for i in range(len(vlines)):
                t = vlines[i]['pos']
                vlines[i]['pos'] = (t[0] + start, t[1] + start)
            if len(vlines) == 1:
                break
        if len(vlines) != 1:
            print('Linhas verticais não encontradas (0/1)')
            print('essay: ' + e)
            return {}
        thresh.append(t)

        LINES[e]['vertical'] += vlines
        LINES[e]['vertical'].sort(key=lambda a: a['pos'][0])

        ehlines = [h['pos'] for h in LINES[e]['horizontal']]
        ehlines.sort()

        # procura as linhas horizontais de uma em uma
        pixels_per_line = (ehlines[1][0] - ehlines[0][1]) / 30
        cropH = int(pixels_per_line / 4)

        for l in range(29):
            for t in np.arange(0.4, 0.9, 0.1):
                start = ehlines[0][1] + int(l * pixels_per_line) + cropH
                end = start + int(pixels_per_line)  # - cropH
                # hlines = horizontalLines(gray[start:end,evlines[0][1]+cropV:vlines[0]['pos'][1]], t, 1, sort='pos', reverse=True) # Procura somente na área da numeração
                hlines = horizontalLines(gray[start:end, :], t, 1, sort='pos', reverse=True)  # Procura em toda a área
                hlines.sort(key=lambda a: a['pos'], reverse=True)
                hlines = hlines[:1]
                for i in range(len(hlines)):
                    t = hlines[i]['pos']
                    hlines[i]['pos'] = (t[0] + start, t[1] + start)
                if len(hlines) == 1:
                    break
            if len(hlines) != 1:
                print('Linhas horizontais não encontradas (' + str(l) + '/' + '29)')
                print('essay: ' + e)
                # cv2_imshow(gray)
                # LINES[e] = {}
                # error.append()
                break
            thresh.append(t)
            LINES[e]['horizontal'] += hlines
        LINES[e]['horizontal'].sort(key=lambda a: a['pos'][0])

    return LINES


# Utilizar com imagem binarizada
# Deixando somente as àreas detectadas
def textLinesBounds(essay, horizontalLines, verticalLines):
    horizontalLines = [h['pos'] for h in horizontalLines]
    verticalLines = [v['pos'] for v in verticalLines]
    # Com a imagem invertida, a soma de uma área totalmente branca será 0
    essay = 255 - essay[:, verticalLines[1][1]:verticalLines[2][0]]
    tlBounds = []
    for i in range(len(horizontalLines)-2):
        # Caso não haja uma divisão exata (linha totalmente branca), o limite inferior englobará a metade da próxima linha
        lowestBound = horizontalLines[i+1][1] + int((horizontalLines[i+2][0] - horizontalLines[i+1][1]) / 2)
        lb = lowestBound
        for y in range(lb, lowestBound+1):
            ys = np.sum(essay[y,:])
            if ys == 0: # tudo branco
                print('HEY')
                lb = y
                break
        tlBounds.append((horizontalLines[i][1], lb))
    # Última linha é exceção
    tlBounds.append((horizontalLines[-2][1], horizontalLines[-1][0]))
    return tlBounds

# ==========================================================================================

def remove_lines_horizontal(img, i):
    if (i):
        kernel = np.array([[-2, 4, -2], [-4, 7.55, -4], [-2, 4, -2]])
    else:
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    leftBorder = cv2.filter2D(img, -1, kernel)
    rightBorder = cv2.filter2D(img, -1, -kernel)

    borders = np.bitwise_or(leftBorder, rightBorder)
    pivot = 128
    borders[borders < pivot] = 0
    borders[borders >= pivot] = 255

    kernel2 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    dilate = cv2.dilate(borders, kernel2, iterations=1)
    # erode = cv2.erode(dilate, kernel2, iterations=1)

    # dilate = cv2.dilate(borders, kernel2, iterations=1)
    # dilate = cv2.dilate(borders, kernel2, iterations=1)
    # erode = cv2.erode(dilate, kernel2, iterations=1)
    return dilate


def remove_lines_vertical(img, i):

    if (i):
        kernel = np.array([[-2, -4, -2], [4, 7.55, 4], [-2, -4, -2]])
    else:
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    topBorder = cv2.filter2D(img, -1, kernel)
    bottomBorder = cv2.filter2D(img, -1, -kernel)

    borders = np.bitwise_or(topBorder, bottomBorder)

    pivot = 100
    borders[borders < pivot] = 0
    borders[borders >= pivot] = 255

    kernel2 = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)

    dilate = cv2.dilate(borders, kernel2, iterations=1)

    erode = cv2.erode(dilate, kernel2, iterations=1)

    # res = np.bitwise_and(dilate, img)

    # res = cv2.dilate(erode, kernel2, iterations=1)

    return erode


def bining_image(img, pivot=255):
    """
    Thresholds a copy of an image using a pivot as a paarmeter
    :param img: Image
    :param pivot: Pixel value that determines how the image is going to be limiarized
    :return: Binarized image
    """
    bining = img.copy()
    bining[bining < pivot] = 0
    bining[bining >= pivot] = 255
    return bining


def remove_lines(img, iterations=10):
    #return remove_lines_vertical(remove_lines_horizontal(img))
    #img = cv2.imread('captura.PNG', 0)
    #cv2_imshow(img)
    pivot = 160
    binimg = bining_image(img, pivot=pivot)
    kernel=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    new_img = remove_lines_horizontal(binimg, True)
    new_img = remove_lines_vertical(new_img, True)
    new_img = cv2.GaussianBlur(new_img, (3, 3), 5)
    pivot = 32
    #new_img[new_img < pivot] = 0
    #new_img[new_img >= pivot] = 255
    new_img = bining_image(new_img, pivot=pivot)
    new_img = cv2.erode(new_img, kernel, iterations=1)
    for i in range(iterations):
        new_img = cv2.GaussianBlur(new_img, (5, 5), 5)
        pivot = 32
        # new_img[new_img < pivot] = 0
        # new_img[new_img >= pivot] = 255
        new_img = bining_image(new_img, pivot=pivot)
        new_img = cv2.erode(new_img, kernel, iterations=1)
        new_img = np.bitwise_and(new_img, binimg)

    return new_img


def show_textLines(essay, LINES, verbose=False):
    """
    Divide essay image by line, in the text rectangle destined to the handwriten text of the candidate
    and returns a list of the lines in the essays that were segmented
    :param essay:
    :param LINES:
    :return:
    """
    binary = bining_image(essay, pivot=64)
    # cv2_imshow(binary)
    bounds = textLinesBounds(binary, LINES['horizontal'], LINES['vertical'])
    vlines = [v['pos'] for v in LINES['vertical']]
    # for i in range(len(bounds)):
    list_of_essay_lines = []
    for i in range(len(bounds)):
        if verbose:
            print('[INFO] Segmetando Linha ' + str(i + 1))
        essay_line = remove_lines(255 - essay[bounds[i][0]:bounds[i][1], vlines[1][1]:vlines[2][1]])
        list_of_essay_lines.append(essay_line)

    return list_of_essay_lines
        # remove_lines(255-essay[bounds[i][0]:bounds[i][1],vlines[1][1]:vlines[2][1]])


def get_labeled_connected_components(img, connectivity=8):
    """
    Takes a binarized image, calculates its connected components and creates a new image with connected components
    labeled by color
    :param img: Image
    :param connectivity: Connectivity level to be used in the connected components
    :return:
    """
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=connectivity)
    label_hue = np.uint8(100 * ret * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    return labeled_img
