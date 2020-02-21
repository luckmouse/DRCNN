from fcm import FCM
import numpy as np

def hcluster(pix_vec, im_di):
    print('... ... 1st round clustering ... ...')
    fcm = FCM(n_clusters=2)
    fcm.fit(pix_vec)
    fcm_lab = fcm.u.argmax(axis=1)

    # 变化类像素数目的上下界
    if sum(fcm_lab==0)<sum(fcm_lab==1):
        ttr = round(sum(fcm_lab==0)*1.25)
        ttl = round(sum(fcm_lab==0)/1.10)
    else:
        ttr = round(sum(fcm_lab==1)*1.25)
        ttl = round(sum(fcm_lab==1)/1.10)

    print('... ... 2nd round clustering ... ...')
    fcm = FCM(n_clusters=5)
    fcm.fit(pix_vec)
    fcm_lab  = fcm.u.argmax(axis=1)

    idx = []
    idx_tmp = []
    idxmean = []
    res_lab = np.zeros(ylen*xlen, dtype=np.float32)
    for i in range(0, 5):
        idx_tmp.append(np.argwhere(fcm_lab==i))
        idxmean.append(im_di.reshape(ylen*xlen, 1)[idx_tmp[i]].mean())

    idx_sort = np.argsort(idxmean)
    for i in range(0, 5):
        idx.append(idx_tmp[idx_sort[i]])

    print('ttl : ', ttl)
    print('ttr : ', ttr)
    print(len(idx[0]), idxmean[idx_sort[0]])
    print(len(idx[1]), idxmean[idx_sort[1]])
    print(len(idx[2]), idxmean[idx_sort[2]])
    print(len(idx[3]), idxmean[idx_sort[3]])
    print(len(idx[4]), idxmean[idx_sort[4]])

    c = len(idx[4])
    res_lab[idx[4]] = 1
    flag_mid = 0
    for i in range(1, 5):
        c = c+len(idx[4-i])
        if c < ttl:
            res_lab[idx[4-i]] = 1
        elif c >= ttl and c < ttr:
            res_lab[idx[4-i]] = 0.5
            flag_mid = 1
        elif flag_mid == 0:
            res_lab[idx[4-i]] = 0.5
            flag_mid = 1
        else:
            res_lab[idx[4-i]] = 0

    res_lab = res_lab.reshape(ylen, xlen)
    return res_lab
