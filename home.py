import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon, QDesktopServices
from PyQt5.QtCore import Qt, QUrl


class HomePage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AP4FED - Home Page")
        self.resize(800, 600)  # Dimensioni iniziali della finestra

        # Layout principale
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

        # Aggiungi il logo
        logo_path = "img/logo.png"  # Percorso del logo
        logo_label = QLabel(self)
        pixmap = QPixmap(logo_path).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # Aggiungi la scritta al centro
        description_label = QLabel("A Lightweight Federated Learning Engine and Benchmark")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(description_label)

        # Aggiungi i pulsanti
        button_start = QPushButton("Start a new project")
        button_start.setStyleSheet("""
            QPushButton {
                background-color: green; 
                color: white; 
                font-size: 14px; 
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00b300;
            }
            QPushButton:pressed {
                background-color: #008000;
            }
        """)
        button_start.setCursor(Qt.PointingHandCursor)  # Imposta la manina
        button_start.clicked.connect(self.start_new_project)

        button_close = QPushButton("Close")
        button_close.setStyleSheet("""
            QPushButton {
                background-color: red; 
                color: white; 
                font-size: 14px; 
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc0000;
            }
        """)
        button_close.setCursor(Qt.PointingHandCursor)  # Imposta la manina
        button_close.clicked.connect(self.close_application)

        layout.addWidget(button_start)
        layout.addWidget(button_close)

        # Layout per la scritta versione e il bottone GitHub
        footer_layout = QHBoxLayout()
        footer_layout.setAlignment(Qt.AlignCenter)

        # Aggiungi la scritta della versione
        version_label = QLabel("1.0.0 version")
        version_label.setStyleSheet("font-size: 12px; color: black; margin: 5px;")
        footer_layout.addWidget(version_label)

        # Aggiungi il bottone GitHub
        github_button = QPushButton()
        github_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                margin-left: 10px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        # Converti QPixmap in QIcon
        github_pixmap = QPixmap("img/github.png").scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        github_icon = QIcon(github_pixmap)  # Conversione QPixmap -> QIcon
        github_button.setIcon(github_icon)
        github_button.setCursor(Qt.PointingHandCursor)
        github_button.clicked.connect(self.open_github_link)
        footer_layout.addWidget(github_button)

        # Footer allineato al fondo della finestra
        layout.addStretch()
        layout.addLayout(footer_layout)

        # Personalizza la finestra
        self.setStyleSheet("""
            QWidget {
                background-color: #f4f4f4;
            }
            QLabel {
                margin: 10px;
            }
        """)

    def start_new_project(self):
        self.second_screen = SecondScreen()
        self.second_screen.show()

    def close_application(self):
        # Mostra un popup di conferma
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Confirmation")
        msg_box.setText("Are you sure you want to close the application?")
        msg_box.setIcon(QMessageBox.Question)

        # Imposta il font nero per il testo
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                color: black;
                font-size: 14px;
            }
            QMessageBox QPushButton {
                background-color: lightgray;
                color: black;
                font-size: 12px;
                padding: 5px;
                border-radius: 5px;
            }
            QMessageBox QPushButton:hover {
                background-color: gray;
                color: white;
            }
        """)

        # Bottoni personalizzati
        yes_button = msg_box.addButton("Yes", QMessageBox.YesRole)
        no_button = msg_box.addButton("No", QMessageBox.NoRole)
        yes_button.setCursor(Qt.PointingHandCursor)  # Manina per "Yes"
        no_button.setCursor(Qt.PointingHandCursor)  # Manina per "No"
        yes_button.setStyleSheet("background-color: green; color: white; font-size: 12px; padding: 5px;")
        no_button.setStyleSheet("background-color: red; color: white; font-size: 12px; padding: 5px;")

        msg_box.exec_()

        # Chiusura solo se si clicca "Yes"
        if msg_box.clickedButton() == yes_button:
            self.close()

    def open_github_link(self):
        # Reindirizza al link GitHub
        QDesktopServices.openUrl(QUrl("https://github.com/IvanComp/AP4Fed"))


class SecondScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AP4FED - New Project")
        self.resize(800, 600)

        # Layout principale
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(main_layout)

        # Layout per i bottoni
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        # Bottone per Docker Compose
        docker_button = QPushButton("Create project using Docker Compose")
        docker_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                font-size: 14px;
                padding: 15px;
                border: 2px solid black;
                border-radius: 10px;
                width: 250px;
                height: 150px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        docker_button.setCursor(Qt.PointingHandCursor)
        button_layout.addWidget(docker_button)

        # Bottone per Local
        local_button = QPushButton("Create project Locally")
        local_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                font-size: 14px;
                padding: 15px;
                border: 2px solid black;
                border-radius: 10px;
                width: 250px;
                height: 150px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        local_button.setCursor(Qt.PointingHandCursor)
        button_layout.addWidget(local_button)

        main_layout.addLayout(button_layout)

        # Personalizza la finestra
        self.setStyleSheet("background-color: white;")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HomePage()
    window.show()
    sys.exit(app.exec_())