import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { GameMenuComponent, GameSettings } from './components/game-menu/game-menu.component';
import { PongGameComponent } from './components/pong-game/pong-game.component';

@Component({
  selector: 'app-root',
  imports: [CommonModule, GameMenuComponent, PongGameComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'pong-rl';
  currentView: 'menu' | 'game' = 'menu';
  gameSettings: GameSettings | null = null;

  onStartGame(settings: GameSettings) {
    this.gameSettings = settings;
    this.currentView = 'game';
  }

  onGameEnded() {
    this.currentView = 'menu';
    this.gameSettings = null;
  }
}
