import { Component, ElementRef, ViewChild, OnInit, OnDestroy, AfterViewInit, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import * as tf from '@tensorflow/tfjs';

interface AIAgent {
  id: string;
  name: string;
  model: tf.Sequential;
  targetModel: tf.Sequential;
  memory: Experience[];
  epsilon: number;
  score: number;
  totalReward: number;
  gamesPlayed: number;
  color: string;
}

interface Experience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

interface GameState {
  ballX: number;
  ballY: number;
  ballDx: number;
  ballDy: number;
  paddle1Y: number;
  paddle2Y: number;
  player1Score: number;
  player2Score: number;
}

interface LearningConfig {
  learningRate: number;
  epsilon: number;
  epsilonDecay: number;
  epsilonMin: number;
  batchSize: number;
  memorySize: number;
  targetUpdateFreq: number;
  gameSpeed: number;
}

@Component({
  selector: 'app-ai-learning',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './ai-learning.component.html',
  styleUrls: ['./ai-learning.component.css']
})
export class AiLearningComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('gameCanvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('networkVis', { static: true }) networkVisRef!: ElementRef<HTMLDivElement>;
  @Output() backToMenu = new EventEmitter<void>();

  private ctx!: CanvasRenderingContext2D;
  private animationId!: number;
  isTraining = false;
  gameNumber = 0; // Made public
  episode = 0;    // Made public

  // Game configuration
  canvas = { width: 800, height: 400 };
  ball = { x: 400, y: 200, dx: 4, dy: 3, radius: 8 };
  paddle1 = { x: 20, y: 175, width: 10, height: 80 };
  paddle2 = { x: 770, y: 175, width: 10, height: 80 };
  
  // AI Agents
  agent1: AIAgent | null = null;
  agent2: AIAgent | null = null;

  // Learning configuration
  config: LearningConfig = {
    learningRate: 0.001,
    epsilon: 1.0,
    epsilonDecay: 0.995,
    epsilonMin: 0.01,
    batchSize: 32,
    memorySize: 10000,
    targetUpdateFreq: 100,
    gameSpeed: 1
  };

  // Training statistics
  stats = {
    agent1Wins: 0,
    agent2Wins: 0,
    totalGames: 0,
    avgRewardAgent1: 0,
    avgRewardAgent2: 0,
    agent1WinRate: 0,
    agent2WinRate: 0
  };

  // Visualization
  private readonly lossHistory: number[] = [];
  private readonly rewardHistory: { agent1: number[], agent2: number[] } = { agent1: [], agent2: [] };
  private readonly winRateHistory: { agent1: number[], agent2: number[] } = { agent1: [], agent2: [] };

  ngOnInit() {
    this.initializeCanvas();
  }

  ngAfterViewInit() {
    this.setupVisualization();
  }

  ngOnDestroy() {
    this.stopTraining();
    this.cleanup();
  }

  private initializeCanvas() {
    const canvas = this.canvasRef.nativeElement;
    canvas.width = this.canvas.width;
    canvas.height = this.canvas.height;
    this.ctx = canvas.getContext('2d')!;
  }

  private async setupVisualization() {
    // Initialize TensorFlow.js visualization
    await tf.ready();
    console.log('TensorFlow.js ready for AI training!');
  }

  private createNeuralNetwork(): tf.Sequential {
    const model = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [8], units: 128, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 32, activation: 'relu' }),
        tf.layers.dense({ units: 3, activation: 'linear' }) // 3 actions: up, stay, down
      ]
    });

    model.compile({
      optimizer: tf.train.adam(this.config.learningRate),
      loss: 'meanSquaredError'
    });

    return model;
  }

  async startTraining() {
    if (this.isTraining) return;

    this.isTraining = true;
    this.gameNumber = 0;
    this.episode = 0;

    // Initialize AI agents
    this.agent1 = {
      id: 'agent1',
      name: 'Red AI',
      model: this.createNeuralNetwork(),
      targetModel: this.createNeuralNetwork(),
      memory: [],
      epsilon: this.config.epsilon,
      score: 0,
      totalReward: 0,
      gamesPlayed: 0,
      color: '#ff4444'
    };

    this.agent2 = {
      id: 'agent2',
      name: 'Blue AI',
      model: this.createNeuralNetwork(),
      targetModel: this.createNeuralNetwork(),
      memory: [],
      epsilon: this.config.epsilon,
      score: 0,
      totalReward: 0,
      gamesPlayed: 0,
      color: '#4444ff'
    };

    // Copy initial weights to target networks
    this.agent1.targetModel.setWeights(this.agent1.model.getWeights());
    this.agent2.targetModel.setWeights(this.agent2.model.getWeights());

    this.resetGame();
    this.trainingLoop();
  }

  stopTraining() {
    this.isTraining = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
  }

  private async trainingLoop() {
    if (!this.isTraining || !this.agent1 || !this.agent2) return;

    // Game simulation step
    await this.gameStep();

    // Update visualization every 10 games
    if (this.gameNumber % 10 === 0) {
      this.updateVisualization();
    }

    // Continue training
    this.animationId = requestAnimationFrame(() => this.trainingLoop());
  }

  private async gameStep() {
    if (!this.agent1 || !this.agent2) return;

    // Get current state
    const state = this.getGameState();

    // Get actions from both agents
    const action1 = await this.getAction(this.agent1, state);
    const action2 = await this.getAction(this.agent2, state);

    // Store previous state
    const prevState = [...state];

    // Execute actions
    this.executeAction(1, action1);
    this.executeAction(2, action2);

    // Update game physics
    this.updateBall();
    const gameResult = this.checkGameEnd();

    // Get new state
    const newState = this.getGameState();

    // Calculate rewards
    const rewards = this.calculateRewards(gameResult);

    // Store experiences
    this.storeExperience(this.agent1, prevState, action1, rewards.agent1, newState, gameResult.gameEnded);
    this.storeExperience(this.agent2, prevState, action2, rewards.agent2, newState, gameResult.gameEnded);

    // Train networks
    if (this.agent1.memory.length >= this.config.batchSize) {
      await this.trainAgent(this.agent1);
    }
    if (this.agent2.memory.length >= this.config.batchSize) {
      await this.trainAgent(this.agent2);
    }

    // Update target networks periodically
    this.episode++;
    if (this.episode % this.config.targetUpdateFreq === 0) {
      this.updateTargetNetworks();
    }

    // Handle game end
    if (gameResult.gameEnded) {
      this.handleGameEnd(gameResult.winner);
      this.resetGame();
      this.gameNumber++;
    }

    // Render game
    this.render();
  }

  private getGameState(): number[] {
    // Normalize all values to [-1, 1] range
    return [
      (this.ball.x / this.canvas.width) * 2 - 1,
      (this.ball.y / this.canvas.height) * 2 - 1,
      this.ball.dx / 10,
      this.ball.dy / 10,
      (this.paddle1.y / this.canvas.height) * 2 - 1,
      (this.paddle2.y / this.canvas.height) * 2 - 1,
      (this.ball.x - this.paddle1.x) / this.canvas.width,
      (this.ball.x - this.paddle2.x) / this.canvas.width
    ];
  }

  private async getAction(agent: AIAgent, state: number[]): Promise<number> {
    // Epsilon-greedy action selection
    if (Math.random() < agent.epsilon) {
      return Math.floor(Math.random() * 3); // Random action: 0, 1, or 2
    }

    // Predict Q-values
    const stateTensor = tf.tensor2d([state]);
    const qValues = agent.model.predict(stateTensor) as tf.Tensor;
    const qArray = await qValues.data();
    stateTensor.dispose();
    qValues.dispose();

    // Return action with highest Q-value
    return qArray.indexOf(Math.max(...Array.from(qArray)));
  }

  private executeAction(player: number, action: number) {
    const paddle = player === 1 ? this.paddle1 : this.paddle2;
    const speed = 8;

    switch (action) {
      case 0: // Move up
        paddle.y = Math.max(0, paddle.y - speed);
        break;
      case 1: // Stay
        break;
      case 2: // Move down
        paddle.y = Math.min(this.canvas.height - paddle.height, paddle.y + speed);
        break;
    }
  }

  private updateBall() {
    this.ball.x += this.ball.dx * this.config.gameSpeed;
    this.ball.y += this.ball.dy * this.config.gameSpeed;

    // Ball collision with top and bottom walls
    if (this.ball.y - this.ball.radius <= 0 || this.ball.y + this.ball.radius >= this.canvas.height) {
      this.ball.dy = -this.ball.dy;
    }

    // Ball collision with paddles
    if (this.ballCollidesWithPaddle(this.paddle1) || this.ballCollidesWithPaddle(this.paddle2)) {
      this.ball.dx = -this.ball.dx;
      // Add some randomness to make it more interesting
      this.ball.dy += (Math.random() - 0.5) * 2;
    }
  }

  private ballCollidesWithPaddle(paddle: any): boolean {
    return this.ball.x - this.ball.radius <= paddle.x + paddle.width &&
           this.ball.x + this.ball.radius >= paddle.x &&
           this.ball.y - this.ball.radius <= paddle.y + paddle.height &&
           this.ball.y + this.ball.radius >= paddle.y;
  }

  private checkGameEnd(): { gameEnded: boolean, winner: number | null } {
    if (this.ball.x < 0) {
      return { gameEnded: true, winner: 2 }; // Agent 2 wins
    } else if (this.ball.x > this.canvas.width) {
      return { gameEnded: true, winner: 1 }; // Agent 1 wins
    }
    return { gameEnded: false, winner: null };
  }

  private calculateRewards(gameResult: { gameEnded: boolean, winner: number | null }) {
    let agent1Reward = 0;
    let agent2Reward = 0;

    if (gameResult.gameEnded) {
      if (gameResult.winner === 1) {
        agent1Reward = 10; // Win reward
        agent2Reward = -10; // Loss penalty
      } else if (gameResult.winner === 2) {
        agent1Reward = -10; // Loss penalty
        agent2Reward = 10; // Win reward
      }
    } else {
      // Small rewards for keeping the ball in play
      const ballDistanceFromCenter = Math.abs(this.ball.y - this.canvas.height / 2);
      const normalizedDistance = ballDistanceFromCenter / (this.canvas.height / 2);
      agent1Reward = 0.1 * (1 - normalizedDistance);
      agent2Reward = 0.1 * (1 - normalizedDistance);
    }

    return { agent1: agent1Reward, agent2: agent2Reward };
  }

  private storeExperience(agent: AIAgent, state: number[], action: number, reward: number, nextState: number[], done: boolean) {
    agent.memory.push({ state, action, reward, nextState, done });
    agent.totalReward += reward;

    // Limit memory size
    if (agent.memory.length > this.config.memorySize) {
      agent.memory.shift();
    }
  }

  private async trainAgent(agent: AIAgent) {
    if (agent.memory.length < this.config.batchSize) return;

    // Sample random batch
    const batch = this.sampleBatch(agent.memory, this.config.batchSize);
    
    const states = batch.map(exp => exp.state);
    const nextStates = batch.map(exp => exp.nextState);

    // Get current Q-values
    const currentQs = agent.model.predict(tf.tensor2d(states)) as tf.Tensor;
    
    // Get next Q-values from target network
    const nextQs = agent.targetModel.predict(tf.tensor2d(nextStates)) as tf.Tensor;
    
    const currentQsArray = await currentQs.data();
    const nextQsArray = await nextQs.data();

    // Prepare training data
    const trainX: number[][] = [];
    const trainY: number[][] = [];

    for (let i = 0; i < batch.length; i++) {
      const exp = batch[i];
      const currentQ = Array.from(currentQsArray.slice(i * 3, (i + 1) * 3));
      const nextQ = Array.from(nextQsArray.slice(i * 3, (i + 1) * 3));
      
      const targetQ = exp.reward;
      if (!exp.done) {
        // Q-learning update rule
        currentQ[exp.action] = targetQ + 0.95 * Math.max(...nextQ);
      } else {
        currentQ[exp.action] = targetQ;
      }
      
      trainX.push(exp.state);
      trainY.push(currentQ);
    }

    // Train the model
    const xs = tf.tensor2d(trainX);
    const ys = tf.tensor2d(trainY);
    
    await agent.model.fit(xs, ys, { epochs: 1, verbose: 0 });

    // Cleanup tensors
    currentQs.dispose();
    nextQs.dispose();
    xs.dispose();
    ys.dispose();

    // Decay epsilon
    agent.epsilon = Math.max(this.config.epsilonMin, agent.epsilon * this.config.epsilonDecay);
  }

  private sampleBatch(memory: Experience[], batchSize: number): Experience[] {
    const batch: Experience[] = [];
    for (let i = 0; i < batchSize; i++) {
      const randomIndex = Math.floor(Math.random() * memory.length);
      batch.push(memory[randomIndex]);
    }
    return batch;
  }

  private updateTargetNetworks() {
    if (this.agent1 && this.agent2) {
      this.agent1.targetModel.setWeights(this.agent1.model.getWeights());
      this.agent2.targetModel.setWeights(this.agent2.model.getWeights());
    }
  }

  private handleGameEnd(winner: number | null) {
    if (!this.agent1 || !this.agent2) return;

    this.agent1.gamesPlayed++;
    this.agent2.gamesPlayed++;

    if (winner === 1) {
      this.agent1.score++;
      this.stats.agent1Wins++;
    } else if (winner === 2) {
      this.agent2.score++;
      this.stats.agent2Wins++;
    }

    this.stats.totalGames++;
    this.updateStats();
  }

  private updateStats() {
    if (this.stats.totalGames > 0) {
      this.stats.agent1WinRate = (this.stats.agent1Wins / this.stats.totalGames) * 100;
      this.stats.agent2WinRate = (this.stats.agent2Wins / this.stats.totalGames) * 100;
    }

    if (this.agent1 && this.agent2) {
      this.stats.avgRewardAgent1 = this.agent1.totalReward / Math.max(1, this.agent1.gamesPlayed);
      this.stats.avgRewardAgent2 = this.agent2.totalReward / Math.max(1, this.agent2.gamesPlayed);
    }
  }

  private resetGame() {
    this.ball.x = this.canvas.width / 2;
    this.ball.y = this.canvas.height / 2;
    this.ball.dx = (Math.random() > 0.5 ? 1 : -1) * 4;
    this.ball.dy = (Math.random() - 0.5) * 6;
    
    this.paddle1.y = (this.canvas.height - this.paddle1.height) / 2;
    this.paddle2.y = (this.canvas.height - this.paddle2.height) / 2;
  }

  private updateVisualization() {
    // Update reward history
    if (this.agent1 && this.agent2) {
      this.rewardHistory.agent1.push(this.agent1.totalReward);
      this.rewardHistory.agent2.push(this.agent2.totalReward);
      
      this.winRateHistory.agent1.push(this.stats.agent1WinRate);
      this.winRateHistory.agent2.push(this.stats.agent2WinRate);
      
      // Keep only last 100 data points
      if (this.rewardHistory.agent1.length > 100) {
        this.rewardHistory.agent1.shift();
        this.rewardHistory.agent2.shift();
        this.winRateHistory.agent1.shift();
        this.winRateHistory.agent2.shift();
      }
    }
  }

  private render() {
    // Clear canvas
    this.ctx.fillStyle = '#0f0f23';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw center line
    this.ctx.strokeStyle = '#00ff00';
    this.ctx.setLineDash([5, 5]);
    this.ctx.beginPath();
    this.ctx.moveTo(this.canvas.width / 2, 0);
    this.ctx.lineTo(this.canvas.width / 2, this.canvas.height);
    this.ctx.stroke();
    this.ctx.setLineDash([]);

    // Draw paddles
    this.ctx.fillStyle = '#ff4444';
    this.ctx.fillRect(this.paddle1.x, this.paddle1.y, this.paddle1.width, this.paddle1.height);
    
    this.ctx.fillStyle = '#4444ff';
    this.ctx.fillRect(this.paddle2.x, this.paddle2.y, this.paddle2.width, this.paddle2.height);

    // Draw ball
    this.ctx.fillStyle = '#ffffff';
    this.ctx.beginPath();
    this.ctx.arc(this.ball.x, this.ball.y, this.ball.radius, 0, Math.PI * 2);
    this.ctx.fill();

    // Draw scores
    this.ctx.fillStyle = '#ffffff';
    this.ctx.font = '24px Courier New';
    this.ctx.textAlign = 'center';
    
    if (this.agent1 && this.agent2) {
      this.ctx.fillText(`${this.agent1.score}`, this.canvas.width / 4, 30);
      this.ctx.fillText(`${this.agent2.score}`, (3 * this.canvas.width) / 4, 30);
    }
  }

  private cleanup() {
    if (this.agent1) {
      this.agent1.model.dispose();
      this.agent1.targetModel.dispose();
    }
    if (this.agent2) {
      this.agent2.model.dispose();
      this.agent2.targetModel.dispose();
    }
  }

  // Configuration methods
  updateLearningRate(value: number) {
    this.config.learningRate = value;
    if (this.agent1 && this.agent2) {
      // Update optimizers with new learning rate
      this.agent1.model.compile({
        optimizer: tf.train.adam(value),
        loss: 'meanSquaredError'
      });
      this.agent2.model.compile({
        optimizer: tf.train.adam(value),
        loss: 'meanSquaredError'
      });
    }
  }

  updateGameSpeed(value: number) {
    this.config.gameSpeed = value;
  }

  updateBatchSize(value: number) {
    this.config.batchSize = value;
  }

  onBackToMenu() {
    this.stopTraining();
    this.backToMenu.emit();
  }

  goBack() {
    this.backToMenu.emit();
  }
}