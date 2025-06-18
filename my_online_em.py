import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict
import os
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split

class Regularizer:
    """Базовый класс для регуляризаторов"""
    def __init__(self, tau, target="phi"):
        self.tau = tau
        self.target = target  # "phi" или "theta"
    
    def compute_gradient(self, matrix):
        raise NotImplementedError

class SparseRegularizer(Regularizer):
    """Регуляризатор разреживания"""
    def compute_gradient(self, matrix):
        return -self.tau * np.sign(matrix)

class DecorrelatorRegularizer(Regularizer):
    """Регуляризатор декорреляции"""
    def compute_gradient(self, matrix):
        if self.target == "phi":
            # Для матрицы phi (темы × слова)
            norms = np.linalg.norm(matrix, axis=0, keepdims=True)
            matrix_norm = matrix / (norms + 1e-10)
            sim_matrix = matrix_norm.T @ matrix_norm
            np.fill_diagonal(sim_matrix, 0)
            return -self.tau * (matrix @ sim_matrix) * 2
        else:
            # Для матрицы theta (документы × темы)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix_norm = matrix / (norms + 1e-10)
            sim_matrix = matrix_norm @ matrix_norm.T
            np.fill_diagonal(sim_matrix, 0)
            return -self.tau * (sim_matrix @ matrix) * 2

# class OnlineEM:  
#     def __init__(self, num_topics=None, j_max=10, vocab_size=None,
#                  gamma=1.0, random_state=42):
#         self.num_topics = num_topics
#         self.j_max = j_max
#         self.vocab_size = vocab_size
#         self.gamma = gamma
#         self.eps = 1e-10
#         self.random_state = random_state
        
#         if random_state is not None:
#             np.random.seed(random_state)
        
#         self.regularizers = []
#         self.phi = np.random.dirichlet(np.ones(num_topics)*0.1, size=vocab_size)
        
#         self.n_wt = np.ones((vocab_size, num_topics)) * 0.001
#         self.n_wt_ = np.zeros((vocab_size, num_topics))
#         self.term_to_idx = None
#         self.idx_to_term = None
#         self.theta = None
#         self.doc_count = 0

#     def add_regularizer(self, regularizer):
#         """Добавляет регуляризатор в модель"""
#         self.regularizers.append(regularizer)

#     def get_prob_matrix_by_counters(self, counters, inplace=False):
#         """
#         :param counters: matrix to normalize rows
#         :param inplace: flag to inplace normalization
#         :return:
#         """
#         if inplace:
#             res = counters
#         else:
#             res = np.copy(counters)

#         res[res < 0] = 0.
#         # set rows where sum of row is small to uniform
#         res[np.sum(res, axis=1) < self.eps, :] = 1.
#         res /= np.sum(res, axis=1)[:, np.newaxis]
#         return res


#     def _apply_regularizers(self, matrix, target):
#         """Применяет регуляризаторы к указанной матрице"""
#         if not self.regularizers:
#             return 0
            
#         grad = np.zeros_like(matrix)
#         for reg in self.regularizers:
#             if reg.target == target:
#                 grad += reg.compute_gradient(matrix)
#         return matrix * grad

#     def init_vocabulary(self, vocabulary):
#         self.term_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
#         self.idx_to_term = {idx: term for idx, term in enumerate(vocabulary)}        
#         self.vocab_size = len(vocabulary)

#     def e_step(self, doc, doc_idx):
#         old_theta = self.theta[:, doc_idx].copy()

#         term_indices = list(doc.keys())
#         n_d = len(term_indices)
        
#         n_dw = np.array(list(doc.values()))
#         n_dw_expanded = np.tile(n_dw[:, np.newaxis], (1, self.num_topics))

#         theta_d = self.theta[:, doc_idx].T
#         theta_expanded = np.tile(theta_d[np.newaxis, :], (n_d, 1))

#         for _ in range(self.j_max):
#             phi_theta = self.phi[term_indices, :] * theta_expanded
#             phi_theta_norm = self.get_prob_matrix_by_counters(phi_theta)

#             n_td = n_dw_expanded * phi_theta_norm
#             sum_n_td = np.sum(n_td, axis=0)

#             # Применяем регуляризаторы к theta
#             reg = self._apply_regularizers(theta_d, "theta")
#             theta_d = np.maximum(sum_n_td + reg, self.eps)
#             theta_d_sums = theta_d.sum(axis=0)
#             theta_d = theta_d / theta_d_sums

#         self.n_wt_[term_indices, :] += n_td
#         self.theta[:, doc_idx] = theta_d.T

#         # Проверяем изменение theta
#         # delta = np.mean(np.abs(self.theta[:, doc_idx] - old_theta))
#         # print(f"Doc {doc_idx}: Δtheta = {delta:.4f}")
        

#     def m_step(self):
#         old_phi = self.phi.copy()

#         self.n_wt = self.gamma * np.maximum(self.n_wt, self.eps) + (1-self.gamma)*np.maximum(self.n_wt_, self.eps)
#         self.n_wt_ = np.zeros((self.vocab_size, self.num_topics))
        
#         # Применяем регуляризаторы к phi
#         reg = self._apply_regularizers(self.phi, "phi")
#         self.phi = self.get_prob_matrix_by_counters((self.n_wt + reg).T).T

#         # delta = np.mean(np.abs(self.phi - old_phi))
#         # print(f"Δphi = {delta:.4f}")

#     def fit_online(self, doc_term_dict, update_every=5):
#         if self.term_to_idx is None:
#             raise ValueError("Vocabulary not initialized. Call init_vocabulary() first.")

#         if self.theta is None:
#             self.theta = np.full((self.num_topics, len(doc_term_dict)), 1/self.num_topics)

#         for new_idx, (doc_id, doc) in enumerate(doc_term_dict.items()):
#             doc_indices = {self.term_to_idx[term]: freq for term, freq in doc.items()}
#             self.e_step(doc_indices, new_idx)
            
#             self.doc_count += 1
#             if self.doc_count % update_every == 0:
#                 self.m_step()

#     def compute_perplexity(self, doc_term_dict):

#         if self.term_to_idx is None:
#             raise ValueError("Vocabulary not initialized. Call init_vocabulary() first.")

#         docs = doc_term_dict.keys()
#         n_dw = np.zeros((self.vocab_size, len(docs)))
#         # self.theta = np.full((self.num_topics, len(docs)), 1/self.num_topics)

#         for doc_idx, doc in enumerate(docs):
#             doc_indices = {self.term_to_idx[w]: cnt for w, cnt in doc_term_dict[doc].items() 
#                           if w in self.term_to_idx}
#             term_indices = np.array(list(doc_indices.keys()))
                
#             n_w = np.array(list(doc_indices.values()))
#             n_dw[term_indices, doc_idx] = n_w.T

#         # # пересчитываем theta для теста
#         # self.theta = np.full((self.num_topics, len(docs)), 1/self.num_topics)
#         # for doc_idx, doc in enumerate(docs):
#         #     doc_indices = {self.term_to_idx[w]: cnt for w, cnt in doc_term_dict[doc].items() 
#         #                   if w in self.term_to_idx}
#         #     self.e_step(doc_indices, doc_idx)

#         p_wd = np.log(self.phi @ self.theta)
#         s1 = np.sum(n_dw * p_wd)
#         total_words_number = np.sum(n_dw)
        
#         return np.exp(-s1 / total_words_number)
        

#     def get_top_words(self, n_words=10):
#         """
#         Возвращает топовые слова для каждой темы
        
#         Args:
#             n_words: количество топовых слов для вывода
            
#         Returns:
#             top_words: словарь {topic_id: [(word, score), ...]}
#         """
#         top_words = {}
#         for topic in range(self.num_topics):
#             # Сортируем слова по убыванию вероятности в теме
#             top_indices = np.argsort(self.phi[:, topic])[::-1][:n_words]
#             topic_words = [
#                 (self.idx_to_term[idx], self.phi[idx, topic]) 
#                 for idx in top_indices
#             ]
#             top_words[topic] = topic_words
#         return top_words

class OnlineEM:  
    def __init__(self, num_topics=None, j_max=10, vocab_size=None,
                 tau0=1024, kappa=0.7, random_state=42):
        self.num_topics = num_topics
        self.j_max = j_max
        self.vocab_size = vocab_size
        self.eps = 1e-10
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        self.tau0 = tau0
        self.kappa = kappa
        self.update_count = 0

        self.regularizers = []
        self.phi = np.random.dirichlet(np.ones(num_topics) * 0.01,size=vocab_size)
        
        self.n_wt = np.ones((vocab_size, num_topics)) * 0.001
        self.n_wt_ = np.zeros((vocab_size, num_topics))
        self.term_to_idx = None
        self.idx_to_term = None
        self.theta = None
        self.doc_count = 0

    def add_regularizer(self, regularizer):
        self.regularizers.append(regularizer)

    def get_prob_matrix_by_counters(self, counters, inplace=False):
        if inplace:
            res = counters
        else:
            res = np.copy(counters)

        res[res < 0] = 0.
        res[np.sum(res, axis=1) < self.eps, :] = 1.
        res /= np.sum(res, axis=1)[:, np.newaxis]
        return res

    def _apply_regularizers(self, matrix, target):
        if not self.regularizers:
            return 0

        grad = np.zeros_like(matrix)
        for reg in self.regularizers:
            if reg.target == target:
                grad += reg.compute_gradient(matrix)
        return matrix * grad

    def init_vocabulary(self, vocabulary):
        self.term_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
        self.idx_to_term = {idx: term for idx, term in enumerate(vocabulary)}        
        self.vocab_size = len(vocabulary)

    def e_step(self, doc, doc_idx):
        old_theta = self.theta[:, doc_idx].copy()

        term_indices = list(doc.keys())
        n_d = len(term_indices)

        n_dw = np.array(list(doc.values()))
        n_dw_expanded = np.tile(n_dw[:, np.newaxis], (1, self.num_topics))

        theta_d = self.theta[:, doc_idx].T
        theta_expanded = np.tile(theta_d[np.newaxis, :], (n_d, 1))

        for _ in range(self.j_max):
            phi_theta = self.phi[term_indices, :] * theta_expanded
            phi_theta_norm = self.get_prob_matrix_by_counters(phi_theta)

            n_td = n_dw_expanded * phi_theta_norm
            sum_n_td = np.sum(n_td, axis=0)

            reg = self._apply_regularizers(theta_d, "theta")
            theta_d = np.maximum(sum_n_td + reg, self.eps)
            theta_d /= theta_d.sum()

        self.n_wt_[term_indices, :] += n_td
        self.theta[:, doc_idx] = theta_d.T

    def m_step(self):
        old_phi = self.phi.copy()

        self.update_count += 1
        gamma_t = (self.tau0 + self.update_count) ** (-self.kappa)

        self.n_wt = (1 - gamma_t) * np.maximum(self.n_wt, self.eps) + gamma_t * np.maximum(self.n_wt_, self.eps)
        self.n_wt_ = np.zeros((self.vocab_size, self.num_topics))

        reg = self._apply_regularizers(self.phi, "phi")
        self.phi = self.get_prob_matrix_by_counters((self.n_wt + reg).T).T

    def fit_online(self, doc_term_dict, update_every=5):
        if self.term_to_idx is None:
            raise ValueError("Vocabulary not initialized. Call init_vocabulary() first.")

        if self.theta is None:
            self.theta = np.full((self.num_topics, len(doc_term_dict)), 1 / self.num_topics)

        for new_idx, (doc_id, doc) in enumerate(doc_term_dict.items()):
            doc_indices = {self.term_to_idx[term]: freq for term, freq in doc.items()}
            self.e_step(doc_indices, new_idx)

            self.doc_count += 1
            if self.doc_count % update_every == 0:
                self.m_step()

    def compute_perplexity(self, doc_term_dict):
        if self.term_to_idx is None:
            raise ValueError("Vocabulary not initialized. Call init_vocabulary() first.")

        docs = doc_term_dict.keys()
        n_dw = np.zeros((self.vocab_size, len(docs)))

        for doc_idx, doc in enumerate(docs):
            doc_indices = {self.term_to_idx[w]: cnt for w, cnt in doc_term_dict[doc].items() 
                           if w in self.term_to_idx}
            term_indices = np.array(list(doc_indices.keys()))
            n_w = np.array(list(doc_indices.values()))
            n_dw[term_indices, doc_idx] = n_w.T

        p_wd = np.log(self.phi @ self.theta)
        s1 = np.sum(n_dw * p_wd)
        total_words_number = np.sum(n_dw)
        
        return np.exp(-s1 / total_words_number)

    def get_top_words(self, n_words=10):
        top_words = {}
        for topic in range(self.num_topics):
            top_indices = np.argsort(self.phi[:, topic])[::-1][:n_words]
            topic_words = [
                (self.idx_to_term[idx], self.phi[idx, topic]) 
                for idx in top_indices
            ]
            top_words[topic] = topic_words
        return top_words

def prepare_data():
    """Загрузка и подготовка данных"""
    print("Загрузка 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes')
    )
    return newsgroups.data

def vectorize_texts(texts, max_features=2000, random_state=None):  
    """Векторизация текстов в BoW формат"""
    print("Векторизация текстов...")

    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=10,
        max_df=0.5,  # Ужесточаем фильтрацию частых слов
        ngram_range=(1, 2),  # Добавляем биграммы
        token_pattern=r'\b[a-zA-Z]{3,}\b',
        analyzer='word'  # Явно указываем анализ по словам
    )
    return vectorizer.fit_transform(texts), vectorizer

def create_term_frequency_dict(X, vectorizer):
    """
    Создает словарь вида {doc_id: {term: frequency}} из матрицы X
    """
    terms = vectorizer.get_feature_names_out()
    doc_term_freq = defaultdict(dict)
    
    rows, cols = X.nonzero()
    for doc_idx, term_idx in zip(rows, cols):
        term = terms[term_idx]
        freq = X[doc_idx, term_idx]
        doc_term_freq[doc_idx][term] = freq
    
    return dict(doc_term_freq)

def print_topic_names(model, n_words=5):
    """
    Печатает названия тем на основе топовых слов
    
    Args:
        model: обученная модель OnlineEM_local_E
        n_words: количество слов для формирования названия темы
    """
    top_words = model.get_top_words(n_words)
    
    print("\nНазвания тем:")
    for topic, words in top_words.items():
        # Берем первые n_words слов и соединяем их через запятую
        topic_name = ", ".join([word for word, score in words[:n_words]])
        print(f"Тема {topic}: {topic_name}")

def print_true_topic_names():
    """Печатает настоящие названия 20 тем из датасета 20 Newsgroups"""
    categories = [
        'alt.atheism',
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'comp.windows.x',
        'misc.forsale',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space',
        'soc.religion.christian',
        'talk.politics.guns',
        'talk.politics.mideast',
        'talk.politics.misc',
        'talk.religion.misc'
    ]
    
    print("\nНастоящие 20 тем в датасете 20 Newsgroups:")
    print("=" * 60)
    for i, category in enumerate(categories, 1):
        print(f"{i:2}. {category.replace('.', ' / ')}")
    
    print("\nПояснения:")
    print("- comp: компьютеры")
    print("- rec: развлечения")
    print("- sci: наука")
    print("- talk: обсуждения")
    print("- alt: альтернативные темы")
    print("- misc: разное")
    print("- soc: социальные темы")

def print_matrix(matrix, xticklabels, yticklabels):
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(20, 10))
        
        # Визуализация
        sns.heatmap(matrix, cmap="YlOrRd", 
                   xticklabels=xticklabels,
                   yticklabels=yticklabels)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Для визуализации установите seaborn и matplotlib")

def main(random_state=42):
    # Установка random state для воспроизводимости
    np.random.seed(random_state)
    
    # 1. Загрузка и подготовка данных
    texts = prepare_data()
    
    # 2. Векторизация
    X, vectorizer = vectorize_texts(texts, random_state=random_state)
    
    # 3. Разделение на train/test (80/20)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=random_state)
    
    # 4. Создание словарей документ-терм-частота
    train_dict = create_term_frequency_dict(X_train, vectorizer)
    test_dict = create_term_frequency_dict(X_test, vectorizer)

    # 5. Инициализация и обучение модели
    model = OnlineEM(
        num_topics=20,
        j_max=15,  # Увеличиваем количество итераций E-шага
        # gamma=0.99,   # Баланс между историей и новыми данными
        vocab_size=len(vectorizer.get_feature_names_out()),
        random_state=42
    )

    # Добавляем регуляризаторы
    # model.add_regularizer(SparseRegularizer(tau=0.5, target="phi"))
    # model.add_regularizer(DecorrelatorRegularizer(tau=0.05, target="phi"))
    # model.add_regularizer(SparseRegularizer(tau=0.5, target="theta"))

    model.init_vocabulary(vectorizer.get_feature_names_out())

    model.fit_online(train_dict, update_every=500)
    
    # 6. Оценка качества
    print("\nОценка качества:")
    print(f"Train perplexity: {model.compute_perplexity(train_dict):.1f}")
    # print(f"Test perplexity: {model.compute_perplexity(test_dict):.1f}")

    # 7. Анализ тем
    print("\nТоповые слова в темах:")
    print_topic_names(model, n_words=10)
    
    print("Минимальное значение:", np.min(model.phi))
    print("Максимальное значение:", np.max(model.phi))
    print("Среднее значение:", np.mean(model.phi))

    # 8. Визуализация матрицы phi
    print("\nВизуализация матрицы phi:")
    num_topics = model.num_topics
    num_words = len(vectorizer.get_feature_names_out())
    xticklabels = [f"{i}" for i in np.arange(num_topics)]
    yticklabels = [f"{i}" for i in np.arange(num_words)]
    # print_matrix(model.phi, xticklabels, yticklabels)

    print("Минимальное значение:", np.min(model.theta))
    print("Максимальное значение:", np.max(model.theta))
    print("Среднее значение:", np.mean(model.theta))

    xticklabels = [f"{i}" for i in np.arange(300)]
    yticklabels = [f"{i}" for i in np.arange(num_topics)]

    print_matrix(model.theta, xticklabels, yticklabels)

if __name__ == "__main__":
    main(random_state=42)